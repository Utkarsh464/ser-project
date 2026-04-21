"""
predict_service.py - LIVE AUDIO OPTIMIZED
Fixes: webm chunks, silence, noise, short clips, normalization mismatch
"""
import numpy as np
import librosa
from tensorflow import keras
import joblib
import warnings
import os
import subprocess
import tempfile
warnings.filterwarnings('ignore')

class EmotionPredictor:

    def __init__(self):
        print("🔄 Loading CNN model...")
        if not os.path.exists('models/emotion_model_cnn.h5'):
            raise FileNotFoundError("❌ Model not found: models/emotion_model_cnn.h5")
        if not os.path.exists('models/label_encoder_cnn.pkl'):
            raise FileNotFoundError("❌ Encoder not found: models/label_encoder_cnn.pkl")

        self.model = keras.models.load_model('models/emotion_model_cnn.h5')
        self.encoder = joblib.load('models/label_encoder_cnn.pkl')
        self.sample_rate = 22050
        self.duration = 3
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512

        # Silence threshold — tune this if needed (0.003 - 0.008)
        self.silence_threshold = 0.004

        print("✅ Model loaded!")
        print(f"   Classes: {list(self.encoder.classes_)}")

    def convert_webm_to_wav(self, input_path):
        """
        Convert webm/any format → clean 22050Hz mono WAV using ffmpeg.
        ffmpeg handles browser audio MUCH better than librosa alone.
        Falls back to librosa if ffmpeg not installed.
        """
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_wav.close()

        try:
            # Try ffmpeg first (best quality conversion)
            result = subprocess.run([
                'ffmpeg', '-y',
                '-i', input_path,
                '-ar', str(self.sample_rate),  # resample to 22050
                '-ac', '1',                     # mono
                '-af', 'aresample=resampler=swr,highpass=f=80,lowpass=f=8000',  # clean audio
                '-loglevel', 'error',
                tmp_wav.name
            ], capture_output=True, timeout=10)

            if result.returncode == 0:
                audio, sr = librosa.load(tmp_wav.name, sr=self.sample_rate, mono=True)
                os.unlink(tmp_wav.name)
                return audio, sr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # ffmpeg not available, fall through to librosa
        except Exception as e:
            print(f"  ffmpeg error: {e}")

        # Fallback: librosa direct load
        try:
            os.unlink(tmp_wav.name)
        except:
            pass
        try:
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            print(f"❌ Audio load failed: {e}")
            return None, None

    def is_silent_or_noisy(self, audio):
        """
        Detect silence AND pure noise (both give bad predictions).
        Returns True if audio is not worth analyzing.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.silence_threshold:
            return True, 'silent'

        # Check if it's mostly noise (flat spectrum = no speech)
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=32)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Speech has high variance across mel bands; noise is flat
        band_variance = np.var(mel_db, axis=1).mean()
        if band_variance < 8.0:
            return True, 'noise'

        return False, 'ok'

    def preprocess_audio(self, audio):
        """
        Match exactly how training data was preprocessed.
        This is the #1 fix for live audio mismatch.
        """
        max_len = self.sample_rate * self.duration

        # Trim leading/trailing silence (common in browser recordings)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        # If trimmed audio is too short, use original
        if len(audio_trimmed) < self.sample_rate * 0.5:
            audio_trimmed = audio

        # Pad or trim to exactly 3 seconds
        audio_trimmed = np.pad(
            audio_trimmed,
            (0, max(0, max_len - len(audio_trimmed)))
        )[:max_len]

        # Normalize amplitude (handles quiet mic input)
        max_val = np.max(np.abs(audio_trimmed))
        if max_val > 0:
            audio_trimmed = audio_trimmed / max_val * 0.9

        return audio_trimmed

    def extract_spectrogram(self, audio):
        """Extract mel spectrogram — identical to training preprocessing."""
        try:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Same normalization as training
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

            return mel_db.astype(np.float32)
        except Exception as e:
            print(f"❌ Spectrogram error: {e}")
            return None

    def predict(self, audio_path):
        """Full prediction pipeline with live audio fixes."""
        try:
            # Step 1: Convert to clean WAV
            audio, sr = self.convert_webm_to_wav(audio_path)
            if audio is None:
                return None, None, None

            # Step 2: Check for silence/noise
            bad, reason = self.is_silent_or_noisy(audio)
            if bad:
                print(f"⚠️  Skipping — {reason}")
                neutral = 'neutral' if 'neutral' in self.encoder.classes_ else self.encoder.classes_[0]
                probs = {e: 0.0 for e in self.encoder.classes_}
                probs[neutral] = 1.0
                return neutral, 1.0, probs

            # Step 3: Preprocess audio
            audio = self.preprocess_audio(audio)

            # Step 4: Extract spectrogram
            spec = self.extract_spectrogram(audio)
            if spec is None:
                return None, None, None

            # Step 5: Predict
            spec_input = spec[np.newaxis, ..., np.newaxis]
            raw_preds = self.model.predict(spec_input, verbose=0)[0]

            # Step 6: Temperature scaling
            # Higher temp = softer predictions = less "stuck on one emotion"
            temperature = 1.8
            scaled = np.exp(np.log(raw_preds + 1e-10) / temperature)
            scaled = scaled / scaled.sum()

            predicted_class = np.argmax(scaled)
            predicted_emotion = self.encoder.inverse_transform([predicted_class])[0]
            confidence = float(scaled[predicted_class])

            all_probs = {
                emotion: float(scaled[i])
                for i, emotion in enumerate(self.encoder.classes_)
            }

            print(f"✅ {predicted_emotion} ({confidence:.1%})  raw_top: {self.encoder.classes_[np.argmax(raw_preds)]} ({raw_preds.max():.1%})")
            return predicted_emotion, confidence, all_probs

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback; traceback.print_exc()
            return None, None, None


predictor = EmotionPredictor()