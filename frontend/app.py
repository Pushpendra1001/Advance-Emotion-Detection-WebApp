from flask import Flask, Response, jsonify, request, send_file, session
from flask_cors import CORS
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import tensorflow as tf
import traceback  # Add traceback import

app = Flask(__name__)
CORS(app, 
     origins=["http://localhost:5173"],  # Single origin instead of multiple
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"])

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


MODEL_PATH = os.path.join('src', 'models', 'emotion_model.keras')


def load_model_file(model_path):
    global model
    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Print detailed model information
        model.summary()
        input_shape = model.input_shape
        print(f"Model expects input shape: {input_shape}")
        
        # Test model with dummy input
        dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("Model test prediction successful")
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def validate_model_path(model_path):
    """Validate and fix model path if necessary"""
    potential_paths = [
        model_path,  
        os.path.join(os.getcwd(), model_path),  
        os.path.join(os.getcwd(), 'models', model_path),  
        os.path.join(os.getcwd(), 'frontend', model_path),  
        os.path.join(os.getcwd(), 'frontend', '/models/emotion_model.keras'),  
        os.path.join(os.getcwd(), 'backend', 'models', '/models/emotion_model.keras')  
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
        # Don't try to access Flask session here since we're outside request context
        if not email or email == 'test@example.com':
            email = 'anonymous'  # Default fallback instead of using session
                
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
            
        # Update session data
        if session_id and session_id in active_sessions:
            if 'emotions' not in active_sessions[session_id]:
                active_sessions[session_id]['emotions'] = []
            active_sessions[session_id]['emotions'].append(new_data)
            
    except Exception as e:
        print(f"Error in save_emotion: {str(e)}")
        traceback.print_exc()
        
        
def process_frame(frame, session_data):
    try:
        if frame is None:
            return frame

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI and preprocess for emotion detection
            face_roi = frame[y:y+h, x:x+w]  # Use color image directly
            face_roi = cv2.resize(face_roi, (96, 96))  # Resize to 96x96 as expected by model
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

            # Make prediction if model is loaded
            if model is not None:
                try:
                    # Get emotion prediction
                    preds = model.predict(face_roi, verbose=0)
                    emotion_idx = np.argmax(preds[0])
                    emotion = emotion_labels[emotion_idx]
                    confidence = float(preds[0][emotion_idx])

                    # Save emotion with session data
                    email = session_data.get('email', 'anonymous')
                    save_emotion(
                        email=email,
                        emotion=emotion,
                        confidence=confidence,
                        model_type=session_data.get('model_type'),
                        session_id=session_data.get('session_id')
                    )

                    # Draw emotion label with improved visibility
                    label = f"{emotion}: {confidence:.2f}"
                    # Draw background rectangle for text
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, 
                                (x, y-30), 
                                (x + label_w, y), 
                                (0, 0, 0), 
                                -1)
                    # Draw text
                    cv2.putText(frame, 
                              label, 
                              (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (255, 255, 255), 
                              2)

                except Exception as e:
                    print(f"Error during prediction: {str(e)}")
                    traceback.print_exc()

        return frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return frame

@app.route('/video_feed')
def video_feed():
    try:
        session_id = request.args.get('session_id')
        print(f"Video feed requested for session: {session_id}")
        print(f"Active sessions: {active_sessions.keys()}")
        
        if not session_id or session_id not in active_sessions:
            return jsonify({"error": "Invalid session"}), 400

        session_data = active_sessions[session_id]
        print(f"Session data: {session_data}")
        
        def generate_frames():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open video capture")

            try:
                while session_id in active_sessions:
                    success, frame = cap.read()
                    if not success:
                        break

                    processed_frame = process_frame(frame, session_data)
                    
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            finally:
                if cap.isOpened():
                    cap.release()

        return Response(generate_frames(),
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
    try:
        data = request.get_json()
        patient_name = data.get('patientName')
        model_type = data.get('modelType')
        
        if not patient_name or not model_type:
            return jsonify({"error": "Missing required fields"}), 400
            
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "patient_name": patient_name,
            "model_type": model_type,
            "start_time": datetime.now().isoformat(),
            "session_id": session_id,
            "email": "test@example.com"  # Or get from actual user session
        }
        
        return jsonify({"sessionId": session_id})
        
    except Exception as e:
        print(f"Error starting session: {str(e)}")
        return jsonify({"error": str(e)}), 500

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