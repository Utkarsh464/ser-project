import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ser-secret-key-change-in-production-2025')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///emotion.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    UPLOAD_FOLDER = 'static/uploads'
    TEMP_AUDIO_FOLDER = 'data/temp_audio'
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm', 'flac', 'm4a'}
    MODEL_PATH = 'models/emotion_model_cnn.h5'
    ENCODER_PATH = 'models/label_encoder_cnn.pkl'
