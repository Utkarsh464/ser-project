"""
Regenerate label_encoder_cnn.pkl from RAVDESS data
Run: python generate_encoder.py
"""
import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# RAVDESS emotion map (3rd number in filename = emotion)
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

RAVDESS_PATH = r'C:\Users\91952\OneDrive\Documents\Desktop\emotion-recognition\data\audio\ravdess'

labels = []
print("Scanning RAVDESS files...")
for actor in os.listdir(RAVDESS_PATH):
    actor_path = os.path.join(RAVDESS_PATH, actor)
    if not os.path.isdir(actor_path):
        continue
    for fname in os.listdir(actor_path):
        if fname.endswith('.wav'):
            parts = fname.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = RAVDESS_EMOTIONS.get(emotion_code)
                if emotion:
                    labels.append(emotion)

print(f"Found {len(labels)} files")
print(f"Emotions found: {set(labels)}")

# Fit encoder on same emotions your model was trained on
label_encoder = LabelEncoder()
label_encoder.fit(sorted(set(labels)))

print(f"Classes: {label_encoder.classes_}")

# Save
save_path = r'E:\projects\ser_project\models\label_encoder_cnn.pkl'
joblib.dump(label_encoder, save_path)
print(f"✅ Saved to: {save_path}")