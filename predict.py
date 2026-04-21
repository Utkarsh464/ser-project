"""
Prediction script for CNN model
"""
import numpy as np
import librosa
from tensorflow import keras
import joblib
import sys
import os

def extract_spectrogram(file_path, sample_rate=22050, duration=3, n_mels=128):
    """Extract mel-spectrogram"""
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    
    max_length = sample_rate * duration
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    
    # Load model and encoder
    print("Loading CNN model...")
    model = keras.models.load_model('models/emotion_model_cnn.h5')
    label_encoder = joblib.load('models/label_encoder_cnn.pkl')
    
    # Extract spectrogram
    print(f"Processing: {audio_path}")
    spec = extract_spectrogram(audio_path)
    spec = spec[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    
    # Predict
    predictions = model.predict(spec, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    confidence = predictions[predicted_class]
    
    # Display results
    print("\n" + "="*60)
    print(f"ðŸŽ¯ PREDICTED EMOTION: {predicted_emotion.upper()}")
    print(f"ðŸ“Š CONFIDENCE: {confidence:.2%}")
    print("="*60)
    
    print("\nðŸ“ˆ All Probabilities:")
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        emotion = label_encoder.classes_[idx]
        prob = predictions[idx]
        bar = "â–ˆ" * int(prob * 50)
        print(f"{emotion:12s} {prob:6.2%} {bar}")
    
    return predicted_emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_cnn.py <audio_file.wav>")
        print("Example: python predict_cnn.py data/audio/happy/sample.wav")
    else:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            predict_emotion(audio_path)
        else:
            print(f"âŒ File not found: {audio_path}")