"""
FIXED Training Script - Better generalization, less fear bias
Run: python train_model.py
"""
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import joblib
import warnings
warnings.filterwarnings('ignore')

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fear', '07': 'disgust', '08': 'surprise'
}

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128

# ── 1. Load RAVDESS (Actor folder format) ───────────────────────────────────
def load_ravdess(data_path):
    file_paths, labels = [], []
    for actor in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if not fname.endswith('.wav'):
                continue
            parts = fname.split('-')
            if len(parts) < 3:
                continue
            emotion = RAVDESS_EMOTIONS.get(parts[2])
            if emotion:
                file_paths.append(os.path.join(actor_path, fname))
                labels.append(emotion)
    print(f"Loaded {len(file_paths)} files")
    from collections import Counter
    print("Distribution:", dict(Counter(labels)))
    return file_paths, labels

# ── 2. Load plain folder format (angry/, happy/, etc.) ─────────────────────
def load_plain(data_path):
    file_paths, labels = [], []
    emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
    for emotion in emotions:
        ep = os.path.join(data_path, emotion)
        if not os.path.exists(ep):
            continue
        for f in os.listdir(ep):
            if f.endswith(('.wav', '.mp3')):
                file_paths.append(os.path.join(ep, f))
                labels.append(emotion)
    return file_paths, labels

# ── 3. Feature extraction with augmentation ─────────────────────────────────
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        max_length = SAMPLE_RATE * DURATION
        audio = np.pad(audio, (0, max(0, max_length - len(audio))))[:max_length]

        if augment:
            choice = np.random.randint(4)
            if choice == 0:   # Add noise
                audio = audio + 0.005 * np.random.randn(len(audio))
            elif choice == 1: # Time stretch
                rate = np.random.uniform(0.85, 1.15)
                audio = librosa.effects.time_stretch(audio, rate=rate)
                audio = np.pad(audio, (0, max(0, max_length - len(audio))))[:max_length]
            elif choice == 2: # Pitch shift
                steps = np.random.randint(-3, 4)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
            # choice==3: no augmentation

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db
    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None

def prepare_dataset(file_paths, labels, augment=False):
    X, y = [], []
    for i, (fp, label) in enumerate(zip(file_paths, labels)):
        if i % 100 == 0:
            print(f"  {i}/{len(file_paths)}")
        spec = extract_features(fp, augment=augment)
        if spec is not None:
            X.append(spec)
            y.append(label)
            if augment:  # Add one augmented copy per sample
                spec2 = extract_features(fp, augment=True)
                if spec2 is not None:
                    X.append(spec2)
                    y.append(label)
    return np.array(X)[..., np.newaxis], np.array(y)

# ── 4. Model ─────────────────────────────────────────────────────────────────
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.30),

        # Block 3
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.35),

        # Dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("FIXED CNN TRAINING — with augmentation & class balancing")
    print("="*60)

    DATA_PATH  = r'C:\Users\91952\OneDrive\Documents\Desktop\emotion-recognition\data\audio\ravdess'
    MODEL_PATH = r'E:\projects\ser_project\models\emotion_model_cnn.h5'
    ENC_PATH   = r'E:\projects\ser_project\models\label_encoder_cnn.pkl'

    # Auto-detect format
    sample = os.listdir(DATA_PATH)[0]
    if sample.startswith('Actor_'):
        print("\n[1] Detected RAVDESS Actor format")
        file_paths, labels = load_ravdess(DATA_PATH)
    else:
        print("\n[1] Detected plain emotion folder format")
        file_paths, labels = load_plain(DATA_PATH)

    if not file_paths:
        print("ERROR: No files found!"); return

    print("\n[2] Extracting features + augmentation (2x data)...")
    X, y = prepare_dataset(file_paths, labels, augment=True)
    print(f"Dataset shape: {X.shape}")

    print("\n[3] Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, ENC_PATH)
    print(f"Classes: {le.classes_}")
    print(f"Encoder saved: {ENC_PATH}")

    print("\n[4] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Class weights to fix imbalance (fear bias fix)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    print("\n[5] Training...")
    model = build_model(X_train.shape[1:], len(le.classes_))
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Encoder saved: {ENC_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()
