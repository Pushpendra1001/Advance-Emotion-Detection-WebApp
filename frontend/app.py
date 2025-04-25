import os
import traceback
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, session
from flask_cors import CORS
import tensorflow as tf
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json


app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SECURE=False,  
    SESSION_COOKIE_SAMESITE='Lax',  
    SESSION_COOKIE_HTTPONLY=True,
    SECRET_KEY='your-secret-key-here'
)


CORS(app, 
     origins=["http://localhost:5173", "http://127.0.0.1:5173"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

USERS_CSV = 'data/users.csv'
EMOTIONS_CSV = 'data/emotions.csv'


os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=['email', 'password']).to_csv(USERS_CSV, index=False)
if not os.path.exists(EMOTIONS_CSV):
    pd.DataFrame(columns=['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']).to_csv(EMOTIONS_CSV, index=False)


model = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['anger', 'Happy', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
active_sessions = {}
active_cameras = {}


MODEL_PATH = os.path.join('src', 'models', 'my_model.keras')  


def load_model_file(model_path):
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def validate_model_path(model_path):
    """Validate and fix model path if necessary"""
    potential_paths = [
        model_path,  
        os.path.join(os.getcwd(), model_path),  
        os.path.join(os.getcwd(), 'models', model_path),  
        os.path.join(os.getcwd(), 'frontend', model_path),  
        os.path.join(os.getcwd(), 'frontend', '/models/my_model.keras'),  
        os.path.join(os.getcwd(), 'backend', 'models', '/models/my_model.keras')  
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.keras'):
                full_path = os.path.join(root, file)
                print(f"Found potential model file: {full_path}")
                return full_path
    
    return None
    
def save_emotion(email, emotion, confidence, model_type, session_id=None):
    try:
        
        if not email or email == 'test@example.com':
            email = session.get('user')
            if not email:
                print("Warning: No valid user email for emotion saving")
                return
                
        timestamp = datetime.now()
        new_data = {
            'email': email,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'emotion': emotion,
            'confidence': float(confidence),
            'model_type': model_type,
            'session_id': session_id
        }
        
        
        try:
            
            os.makedirs(os.path.dirname(EMOTIONS_CSV), exist_ok=True)
            
            
            if os.path.exists(EMOTIONS_CSV):
                df = pd.read_csv(EMOTIONS_CSV)
            else:
                df = pd.DataFrame(columns=['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id'])
            
            
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(EMOTIONS_CSV, index=False)
            print(f"Emotion saved to CSV: {emotion} ({confidence*100:.1f}%)")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            traceback.print_exc()
            
        
        if session_id and session_id in active_sessions:
            if 'emotions' not in active_sessions[session_id]:
                active_sessions[session_id]['emotions'] = []
            
            active_sessions[session_id]['emotions'].append(new_data)
            
    except Exception as e:
        print(f"Error in save_emotion: {str(e)}")
        traceback.print_exc()
        
        
def process_frame(frame, email, model_type, session_id):
    try:
        if model is None:
            print("No model loaded. Please load a model first.")
            return frame
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        
        for (x, y, w, h) in faces:
            
            face_roi = gray[y:y+h, x:x+w]
            
            
            face_roi = cv2.resize(face_roi, (48, 48))  
            face_roi = face_roi.astype('float32') / 255.0  
            
            
            face_roi = np.expand_dims(face_roi, axis=-1)  
            face_roi = np.expand_dims(face_roi, axis=0)   
            
            
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[0][emotion_idx])
            
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion}: {confidence*100:.1f}%"
            
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame, (x, y-30), (x + text_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                       (0, 0, 0), 2)  
            
            
            if session_id and session_id in active_sessions:
                save_emotion(email, emotion, confidence, model_type, session_id)
            
            print(f"Detected emotion: {emotion} ({confidence*100:.1f}%)")
            
        return frame
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return frame

@app.route('/check-models')
def check_models():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    
    model_files = []
    
    
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.keras') or file.endswith('.h5'):
                rel_path = os.path.relpath(os.path.join(root, file), os.getcwd())
                model_files.append(rel_path)
    
    return jsonify({
        "available_models": model_files,
        "cwd": os.getcwd()
    })
    

def generate_frames(email, model_type, session_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")
    
    active_cameras[session_id] = cap
    
    while session_id in active_sessions:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            
            processed_frame = process_frame(frame, email, model_type, session_id)
            
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            continue
    
    if cap.isOpened():
        cap.release()

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        df = pd.read_csv(USERS_CSV)
        if email in df['email'].values:
            return jsonify({"error": "User already exists"}), 400
            
        hashed_password = generate_password_hash(password)
        new_user = pd.DataFrame([{'email': email, 'password': hashed_password}])
        df = pd.concat([df, new_user], ignore_index=True)
        df.to_csv(USERS_CSV, index=False)
        
        session['user'] = email
        return jsonify({"status": "success", "email": email})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        df = pd.read_csv(USERS_CSV)
        user = df[df['email'] == email]
        
        if user.empty or not check_password_hash(user.iloc[0]['password'], password):
            return jsonify({"error": "Invalid credentials"}), 401
            
        
        session['user'] = email
        session.permanent = True  
        
        print(f"User logged in successfully: {email}")
        
        return jsonify({
            "status": "success", 
            "email": email,
            "message": "Login successful"
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        
        email = session.get('user')
        if not email:
            return jsonify({"error": "User not authenticated"}), 401
            
        print(f"Fetching analytics for authenticated user: {email}")
        
        if not os.path.exists(EMOTIONS_CSV):
            print(f"No emotions CSV file found for user: {email}")
            return jsonify({
                "sessionHistory": [],
                "emotionsByTime": [],
                "emotionsByModel": [],
                "emotionTrends": []
            })
        
        
        df = pd.read_csv(EMOTIONS_CSV)
        user_data = df[df['email'] == email].copy()
        
        print(f"Found {len(user_data)} records for user: {email}")
        
        if user_data.empty:
            print(f"No emotion data found for user: {email}")
            return jsonify({
                "sessionHistory": [],
                "emotionsByTime": [],
                "emotionsByModel": [],
                "emotionTrends": []
            })

        
        user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
        
        

        
        session_summaries = []
        for session_id in user_data['session_id'].unique():
            if pd.isna(session_id):
                continue
                
            session_data = user_data[user_data['session_id'] == session_id]
            
            try:
                emotion_counts = session_data['emotion'].value_counts().to_dict()
                total_emotions = len(session_data)
                
                emotion_percentages = {
                    emotion: round((count / total_emotions * 100), 2)
                    for emotion, count in emotion_counts.items()
                }
                
                session_summaries.append({
                    'sessionId': session_id,
                    'startTime': session_data['timestamp'].min().isoformat(),
                    'endTime': session_data['timestamp'].max().isoformat(),
                    'duration': round((session_data['timestamp'].max() - 
                                    session_data['timestamp'].min()).total_seconds() / 60, 2),
                    'modelType': session_data['model_type'].iloc[0],
                    'dominantEmotion': max(emotion_counts.items(), key=lambda x: x[1])[0],
                    'totalDetections': total_emotions,
                    'emotionBreakdown': emotion_counts,
                    'emotionPercentages': emotion_percentages
                })
            except Exception as e:
                print(f"Error processing session {session_id}: {str(e)}")
                continue

        
        session_summaries.sort(key=lambda x: x['startTime'], reverse=True)
        
        
        emotions_by_time = (user_data.groupby([user_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M'), 'emotion'])
            .size()
            .reset_index(name='count')
            .to_dict('records'))
            
        
        emotions_by_model = (user_data.groupby(['model_type', 'emotion'])
            .size()
            .reset_index(name='count')
            .to_dict('records'))
            
        
        emotion_trends = (user_data.groupby([user_data['timestamp'].dt.hour, 'emotion'])
            .size()
            .reset_index(name='count')
            .to_dict('records'))

        return jsonify({
            "sessionHistory": session_summaries,
            "emotionsByTime": emotions_by_time,
            "emotionsByModel": emotions_by_model,
            "emotionTrends": emotion_trends
        })
        
    except Exception as e:
        print(f"Analytics Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({"status": "success"})

@app.route('/check-auth')
def check_auth():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    user = session.get('user')
    if user:
        return jsonify({"status": "authenticated", "email": user})
    return jsonify({"status": "unauthenticated"}), 401

@app.route('/')
def index():
    return jsonify({"status": "running"})


@app.route('/model-status', methods=['GET'])
def model_status():
    try:
        if model is None:
            return jsonify({
                "loaded": False,
                "error": "No model loaded"
            })
        
        return jsonify({
            "loaded": True,
            "model_info": {
                "name": str(model.name) if hasattr(model, 'name') else None,
                "status": "loaded"
            }
        })
    except Exception as e:
        print(f"Error in model-status endpoint: {str(e)}")
        return jsonify({
            "loaded": False,
            "error": str(e)
        }), 500

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    try:
        data = request.get_json()
        model_path = data.get('modelPath')
        
        if not model_path:
            return jsonify({"error": "Model path not provided"}), 400
        
        
        valid_path = validate_model_path(model_path)
        if not valid_path:
            return jsonify({"error": f"Model file not found at {model_path} or any expected location"}), 404
        
        print(f"Validated model path: {valid_path}")
        success = load_model_file(valid_path)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": "Model loaded successfully",
                "path": valid_path
            })
        else:
            return jsonify({"error": "Failed to load model"}), 500
            
    except Exception as e:
        print(f"Exception loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/video_feed')
def video_feed():
    try:
        model_type = request.args.get('model_type', 'general-analysis')
        session_id = request.args.get('session_id')
        email = session.get('user', 'test@example.com')
        
        if model is None:
            return jsonify({"error": "No model loaded"}), 400
            
        def generate():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not start camera.")
                
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                processed_frame = process_frame(frame, email, model_type, session_id)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            cap.release()
            
        return Response(generate(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
                       
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        return jsonify({"error": str(e)}), 500


def create_preview_frame(text):
    
    frame = np.zeros((480, 640, 3), np.uint8)
    
    
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

@app.route('/start-session', methods=['POST'])
def start_session():
    
    
    
    
    
    if model is None:
        return jsonify({"error": "No model loaded. Please load a model first."}), 400
    
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "user": session.get("user", "test@example.com"),  
        "start_time": datetime.now(),
        "emotions": []
    }
    
    return jsonify({"sessionId": session_id})

@app.route('/stop-session', methods=['POST'])
def stop_session():
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        email = data.get('email', session.get('user'))
        
        if not session_id or session_id not in active_sessions:
            return jsonify({"error": "Invalid or expired session"}), 400
            
        
        session_data = active_sessions[session_id]
        session_data['end_time'] = datetime.now()
        
        
        start_time = session_data.get('start_time')
        end_time = session_data['end_time']
        duration = (end_time - start_time).total_seconds() / 60  
        
        emotions = session_data.get('emotions', [])
        total_detections = len(emotions)
        
        
        emotion_counts = {}
        for e in emotions:
            emotion = e['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        
        emotion_percentages = {
            emotion: (count/total_detections * 100) 
            for emotion, count in emotion_counts.items()
        } if total_detections > 0 else {}
        
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        
        
        report = {
            'sessionId': session_id,
            'startTime': start_time.isoformat(),
            'endTime': end_time.isoformat(),
            'duration': duration,
            'totalDetections': total_detections,
            'emotionBreakdown': emotion_counts,
            'emotionPercentages': emotion_percentages,
            'dominantEmotion': dominant_emotion,
            'modelType': session_data.get('model_type')
        }
        
        
        if session_id in active_cameras:
            camera = active_cameras[session_id]
            if camera and camera.isOpened():
                camera.release()
            del active_cameras[session_id]
            
        del active_sessions[session_id]
        
        print(f"Session {session_id} stopped successfully")
        return jsonify(report)
        
    except Exception as e:
        print(f"Error stopping session: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.after_request
def after_request(response):
    
    if request.method == "OPTIONS":
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Max-Age', '3600')
    
    else:
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    return response

@app.route('/status')
def status():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({"status": "running"})

@app.route('/debug-session')
def debug_session():
    return jsonify({
        "session_data": dict(session),
        "user": session.get('user'),
        "cookies": dict(request.cookies)
    })

if __name__ == '__main__':
    
    default_model_paths = [
        os.path.join(os.getcwd(), 'frontend', 'my_model.keras'),
        os.path.join(os.getcwd(), 'my_model.keras'),
        os.path.join(os.getcwd(), 'backend', 'models', 'my_model.keras')
    ]
    
    for path in default_model_paths:
        if os.path.exists(path):
            print(f"Loading default model from: {path}")
            load_model_file(path)
            break
    
    app.run(debug=True, host='0.0.0.0', port=5005)