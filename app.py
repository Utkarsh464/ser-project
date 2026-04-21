from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from functools import wraps
import os
from datetime import datetime
from config import Config
from models import db, User, EmotionHistory
from predict_service import predictor

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_AUDIO_FOLDER'], exist_ok=True)

# ── actual emotion classes from loaded model ──
EMOTIONS = list(predictor.encoder.classes_)

with app.app_context():
    db.create_all()
    print("✅ Database initialized!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ── Auth ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        if not all([username, email, password, confirm]):
            flash('All fields are required', 'danger')
        elif password != confirm:
            flash('Passwords do not match', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
        else:
            u = User(username=username, email=email)
            u.set_password(password)
            db.session.add(u); db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session.permanent = True
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    user   = User.query.get(session['user_id'])
    recent = EmotionHistory.query.filter_by(user_id=user.id).order_by(EmotionHistory.recorded_at.desc()).limit(5).all()
    total  = EmotionHistory.query.filter_by(user_id=user.id).count()
    top    = db.session.query(EmotionHistory.emotion, db.func.count(EmotionHistory.emotion))\
               .filter_by(user_id=user.id).group_by(EmotionHistory.emotion)\
               .order_by(db.func.count(EmotionHistory.emotion).desc()).first()
    most_common = top[0] if top else None
    emotion_counts = {e: EmotionHistory.query.filter_by(user_id=user.id, emotion=e).count() for e in EMOTIONS}
    return render_template('dashboard.html', user=user, recent_emotions=recent,
                           total_predictions=total, most_common=most_common,
                           emotion_counts=emotion_counts, emotions=EMOTIONS)

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route('/live')
@login_required
def live():
    return render_template('live.html', emotions=EMOTIONS)

@app.route('/record')
@login_required
def record():
    return render_template('record.html', emotions=EMOTIONS)

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html', emotions=EMOTIONS)

# ── Predict endpoints ─────────────────────────────────────────────────────────
def _save_and_predict(file, source):
    filename = secure_filename(f"{source}_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
    filepath = os.path.join(app.config['TEMP_AUDIO_FOLDER'], filename)
    file.save(filepath)
    emotion, confidence, all_probs = predictor.predict(filepath)
    if emotion is None:
        return None, None
    rec = EmotionHistory(user_id=session['user_id'], emotion=emotion,
                         confidence=confidence, audio_filename=filename, source=source)
    db.session.add(rec); db.session.commit()
    return emotion, confidence, all_probs, rec

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400
    result = _save_and_predict(request.files['audio'], request.form.get('source', 'record'))
    if result[0] is None:
        return jsonify({'error': 'Failed to process audio'}), 500
    emotion, confidence, all_probs, rec = result
    return jsonify({'success': True, 'emotion': emotion,
                    'confidence': round(confidence*100, 2),
                    'all_probabilities': {k: round(v*100,2) for k,v in all_probs.items()},
                    'recorded_at': rec.recorded_at.strftime('%Y-%m-%d %H:%M:%S')})

@app.route('/predict_live', methods=['POST'])
@login_required
def predict_live():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio chunk'}), 400
    file = request.files['audio']
    save = request.form.get('save', 'false') == 'true'
    filename = secure_filename(f"live_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.webm")
    filepath = os.path.join(app.config['TEMP_AUDIO_FOLDER'], filename)
    file.save(filepath)
    emotion, confidence, all_probs = predictor.predict(filepath)
    if emotion is None:
        return jsonify({'error': 'Failed to process chunk'}), 500
    if save:
        rec = EmotionHistory(user_id=session['user_id'], emotion=emotion,
                             confidence=confidence, audio_filename=filename, source='live')
        db.session.add(rec); db.session.commit()
    try: os.remove(filepath)
    except: pass
    return jsonify({'success': True, 'emotion': emotion,
                    'confidence': round(confidence*100, 2),
                    'all_probabilities': {k: round(v*100,2) for k,v in all_probs.items()}})

@app.route('/predict_upload', methods=['POST'])
@login_required
def predict_upload():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio_file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    result = _save_and_predict(file, 'upload')
    if result[0] is None:
        return jsonify({'error': 'Failed to process file'}), 500
    emotion, confidence, all_probs, rec = result
    return jsonify({'success': True, 'emotion': emotion,
                    'confidence': round(confidence*100, 2),
                    'all_probabilities': {k: round(v*100,2) for k,v in all_probs.items()},
                    'filename': file.filename,
                    'recorded_at': rec.recorded_at.strftime('%Y-%m-%d %H:%M:%S')})

# ── History ───────────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    user = User.query.get(session['user_id'])
    all_emotions = EmotionHistory.query.filter_by(user_id=user.id).order_by(EmotionHistory.recorded_at.desc()).all()
    emotion_stats = {e: EmotionHistory.query.filter_by(user_id=user.id, emotion=e).count() for e in EMOTIONS}
    return render_template('history.html', user=user, emotions=all_emotions,
                           emotion_stats=emotion_stats, emotion_list=EMOTIONS)

@app.route('/delete_history/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    rec = EmotionHistory.query.get_or_404(id)
    if rec.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(rec); db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})
    flash('Record deleted', 'success')
    return redirect(url_for('history'))

@app.route('/clear_all', methods=['GET', 'POST'])
@login_required
def clear_all():
    EmotionHistory.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    flash('All history cleared', 'info')
    return redirect(url_for('history'))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤  SPEECH EMOTION RECOGNITION  —  Final")
    print("="*60)
    print(f"🧠  Emotions: {EMOTIONS}")
    print("📱  Open: http://localhost:5000")
    print("🛑  Press CTRL+C to stop")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
