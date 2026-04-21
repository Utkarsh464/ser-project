# 🎤 VoiceSense — Speech Emotion Recognition (Final)

## Quick Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your model files into models/ folder
#    models/emotion_model_cnn.h5
#    models/label_encoder_cnn.pkl

# 3. Run the app
python app.py

# 4. Open http://localhost:5000
```

## If model files missing — retrain:
```powershell
python train_model.py
# Takes 60-90 mins, auto-saves both model files
```

## Features
- Live Detection (real-time mic analysis)
- Record & Analyze
- Upload audio files
- Full history with charts
- Per-user login/register

## Emotions (auto-detected from your model)
angry · calm · fear · happy · neutral · sad
