import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import tensorflow as tf
import traceback
import csv
from flask import Flask, Response, jsonify, request, send_file, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app, 
     origins=["http://localhost:5173"],
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  
     allow_headers=["Content-Type", "Authorization"])      

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

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  
active_sessions = {}
active_cameras = {}


os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


def find_model_file():
    print("Searching for model file...")
    possible_paths = [
        os.path.join('frontend', 'src', 'models', 'facial_expression_cnn.h5'),
        os.path.join('src', 'models', 'facial_expression_cnn.h5'),
        'facial_expression_cnn.h5',
        os.path.join('models', 'facial_expression_cnn.h5'),
        os.path.join('..', 'models', 'facial_expression_cnn.h5'),
        
        os.path.join('frontend', 'src', 'models', 'emotion_recognition_model_filtered(Angry,happy,neutral).keras'),
        os.path.join('src', 'models', 'emotion_recognition_model_filtered(Angry,happy,neutral).keras'),
        'emotion_recognition_model_filtered(Angry,happy,neutral).keras',
        os.path.join('models', 'emotion_recognition_model_filtered(Angry,happy,neutral).keras'),
        os.path.join('..', 'models', 'emotion_recognition_model_filtered(Angry,happy,neutral).keras')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    
    print("Searching entire directory for facial expression model...")
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if 'facial_expression_cnn' in file and file.endswith('.h5'):
                path = os.path.join(root, file)
                print(f"Found H5 model at: {path}")
                return path
                
    
    print("Searching for any emotion model...")
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if ('emotion' in file.lower() or 'facial' in file.lower()) and (file.endswith('.h5') or file.endswith('.keras')):
                path = os.path.join(root, file)
                print(f"Found model at: {path}")
                return path
                
    return None


def validate_model_labels():
    """Validate that the model's output matches our emotion labels"""
    try:
        global emotion_labels  
        
        if model is None:
            print("No model to validate")
            return
            
        output_shape = model.output_shape
        if output_shape[1] != len(emotion_labels):
            print(f"WARNING: Model outputs {output_shape[1]} classes but we have {len(emotion_labels)} emotion labels!")
            print("This mismatch will cause incorrect emotion labeling.")
            
            if output_shape[1] == 7 and len(emotion_labels) != 7:
                print("Updating emotion labels to match 7-class model")
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            elif output_shape[1] == 3 and len(emotion_labels) != 3:
                print("Updating emotion labels to match 3-class model")
                emotion_labels = ['angry', 'happy', 'neutral']
                
        print(f"Model validated: {output_shape[1]} outputs match {len(emotion_labels)} emotion labels")
    except Exception as e:
        print(f"Error validating model labels: {str(e)}") 

try:
    MODEL_PATH = find_model_file()
    if MODEL_PATH:
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        validate_model_labels()  
    else:
        print("ERROR: Emotion model not found. Emotion detection will not work.")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

def validate_model_path(model_path):
    """Validate and fix model path if necessary"""
    potential_paths = [
        model_path,  
        os.path.join(os.getcwd(), model_path),  
        os.path.join(os.getcwd(), 'models', model_path),  
        os.path.join(os.getcwd(), 'frontend', model_path),  
        os.path.join(os.getcwd(), 'frontend', '/models/emotion_recognition_model_filtered(Angry,happy,neutral).keras'),  
        os.path.join(os.getcwd(), 'backend', 'models', '/models/emotion_recognition_model_filtered(Angry,happy,neutral).keras')  
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
    """Save detected emotion to database"""
    try:
        
        patient_name = email
        if session_id and session_id in active_sessions:
            patient_name = active_sessions[session_id].get('patientName', email)
        else:
            patient_name = email if email != 'anonymous' else 'anonymous'
        
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record = {
            'email': patient_name,  
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence,
            'model_type': model_type,
            'session_id': session_id
        }
        
        
        csv_path = os.path.join('data', 'emotions.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(record)
            
        return True
    except Exception as e:
        print(f"Error saving emotion: {str(e)}")
        return False
        
def process_frame(frame, session_data):
    try:
        if frame is None:
            print("Received empty frame")
            return frame

        
        processed_frame = frame.copy()
        
        
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            try:
                
                if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
                    print(f"Face coordinates out of bounds: x={x}, y={y}, w={w}, h={h}, frame={frame.shape}")
                    continue
                    
                
                face_roi = processed_frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    print("Empty face ROI")
                    continue
                
                
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                
                face_roi_resized = cv2.resize(face_roi_rgb, (48, 48))
                
                
                face_roi_norm = preprocess_face_for_emotion(face_roi_rgb)

                
                face_roi_batch = np.expand_dims(face_roi_norm, axis=0)
                
                
                if model is None:
                    print("Model is not loaded! Cannot predict emotion.")
                    cv2.putText(processed_frame, "Model not loaded", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue
                
                
                model_input_shape = model.input_shape
                print(f"Model input shape: {model_input_shape}")

                
                if model_input_shape[3] == 3 and face_roi_batch.shape[3] != 3:
                    
                    grayscale = np.squeeze(face_roi_batch, axis=3)
                    face_roi_batch = np.stack([grayscale, grayscale, grayscale], axis=-1)
                    print(f"Converted grayscale to RGB: {face_roi_batch.shape}")
                elif model_input_shape[3] == 1 and face_roi_batch.shape[3] != 1:
                    
                    face_roi_batch = np.mean(face_roi_batch, axis=3, keepdims=True)
                    print(f"Converted RGB to grayscale: {face_roi_batch.shape}")

                
                preds = model.predict(face_roi_batch, verbose=0)

                
                preds = balance_emotion_predictions(preds, emotion_labels)
                emotion_idx = np.argmax(preds[0])
                
                
                MIN_CONFIDENCE = 0.40  
                if np.max(preds[0]) < MIN_CONFIDENCE:
                    print(f"Low confidence prediction: {np.max(preds[0]):.2f} - treating as neutral")
                    
                    neutral_idx = emotion_labels.index('neutral') if 'neutral' in emotion_labels else None
                    if neutral_idx is not None:
                        emotion_idx = neutral_idx
                        confidence = float(preds[0][emotion_idx])
                    else:
                        
                        emotion = emotion_labels[emotion_idx]
                        confidence = float(preds[0][emotion_idx])
                        print(f"Using low confidence emotion: {emotion}")
                else:
                    
                    confidence = float(preds[0][emotion_idx])

                
                if emotion_idx < len(emotion_labels):
                    emotion = emotion_labels[emotion_idx]
                else:
                    print(f"Warning: Model predicted class {emotion_idx} but we only have {len(emotion_labels)} labels")
                    emotion = "unknown"
                    confidence = 0.0

                
                emotion_scores = {emotion_labels[i]: float(preds[0][i]) for i in range(len(emotion_labels))}
                print(f"All emotion scores: {emotion_scores}")
                print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
                
                
                patient_name = session_data.get('patientName', 'anonymous')
                
                
                save_emotion(
                    email=patient_name,
                    emotion=emotion,
                    confidence=confidence,
                    model_type=session_data.get('model_type', 'unknown'),
                    session_id=session_data.get('id')
                )
                
                
                label = f"{emotion}: {confidence:.2f}"
                
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(processed_frame, 
                            (x, y-30), 
                            (x + label_w, y), 
                            (0, 0, 0), 
                            -1)
                
                cv2.putText(processed_frame, 
                          label, 
                          (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, 
                          (255, 255, 255), 
                          2)
            
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                traceback.print_exc()
                
        return processed_frame
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        traceback.print_exc()
        return frame

@app.route('/video_feed')
def video_feed():
    try:
        session_id = request.args.get('session_id')
        print(f"Video feed requested for session: {session_id}")
        
        
        if model is not None:
            print("Model input shape:", model.input_shape)
            print("Model output shape:", model.output_shape)
            print("Model expects:", model.input_shape[1:])
            print("Emotion labels:", emotion_labels)
        
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
    """Initialize a new emotion tracking session"""
    try:
        data = request.json
        session_id = str(uuid.uuid4())
        
        
        session = {
            'id': session_id,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patientName': data.get('patientName', 'anonymous'),  
            'model_type': data.get('modelType', 'general'),
            'emotions': []
        }
        
        active_sessions[session_id] = session
        
        return jsonify({
            'success': True,
            'sessionId': session_id
        }), 200
    except Exception as e:
        print(f"Error starting session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop-session', methods=['POST'])
def stop_session():
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        email = data.get('email', session.get('user'))
        
        if not session_id:
            print("Missing session_id in stop-session request")
            return jsonify({"error": "Session ID required"}), 400
            
        if session_id not in active_sessions:
            print(f"Session ID {session_id} not found in active_sessions: {list(active_sessions.keys())}")
            
            return jsonify({
                "sessionId": session_id,
                "status": "stopped",
                "message": "Session already ended or not found",
                "dominantEmotion": "unknown",
                "duration": 0
            }), 200
            
        
        session_data = active_sessions[session_id]
        session_data['end_time'] = datetime.now()
        
        
        start_time = session_data.get('start_time')
        if isinstance(start_time, str):
            try:
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                
                print(f"Could not parse start_time: {start_time}")
                start_time = datetime.now() - timedelta(minutes=1)
        else:
            start_time = datetime.now() - timedelta(minutes=1)
            
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
            'startTime': start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time,
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

@app.route('/emotion-data')
def get_emotion_data():
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({"error": "Invalid session ID"}), 400
            
        session_data = active_sessions[session_id]
        
        
        emotions = []
        if os.path.exists(EMOTIONS_CSV):
            with open(EMOTIONS_CSV, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['session_id'] == session_id:
                        emotions.append({
                            'timestamp': row['timestamp'],
                            'emotion': row['emotion'],
                            'confidence': float(row['confidence'])
                        })
        
        
        session_data['emotions'] = emotions
        
        return jsonify({
            'success': True,
            'emotions': emotions
        })
    except Exception as e:
        print(f"Error getting emotion data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'emotions': []
        }), 500

def load_model_file(model_path):
    """Load the TensorFlow model from disk"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return False
                
        print(f"Loading model from {model_path}")
        global model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return False

def fix_model_input_shape():
    """Attempt to fix model input shape discrepancy"""
    try:
        global model
        if model is None:
            print("No model to fix")
            return False
        
        
        expected_shape = model.layers[0].input_shape
        print(f"Model expects input shape: {expected_shape}")
        
        
        if expected_shape and len(expected_shape) == 4:
            if expected_shape[3] == 1:  
                print("Model expects grayscale input. Updating preprocessing.")
            elif expected_shape[3] == 3:  
                print("Model expects RGB input. Updating preprocessing.")
                
        return True
    except Exception as e:
        print(f"Error fixing model input shape: {str(e)}")
        return False

def inspect_model():
    """Print out detailed model architecture and expected input shape"""
    try:
        if model is None:
            print("No model loaded to inspect")
            return
            
        
        print("===== MODEL DETAILS =====")
        model.summary()
        
        
        input_shape = model.input_shape
        print(f"Model input shape: {input_shape}")
        
        
        first_layer = model.layers[0]
        
        
        
        output_shape = model.layers[-1].output_shape
        
        
        
        if output_shape[1] != len(emotion_labels):
            print(f"WARNING: Model outputs {output_shape[1]} classes but we have {len(emotion_labels)} emotion labels!")
            print("This mismatch could cause incorrect emotion labeling.")
        
        
        if input_shape[1:]:
            
            rgb_input = np.random.random((1, 48, 48, 3)).astype('float32')
            try:
                _ = model.predict(rgb_input, verbose=0)
                print(f"Model successfully processes RGB input with shape {rgb_input.shape}")
            except Exception as e:
                print(f"Model FAILS on RGB input: {e}")
            
            
            gray_input = np.random.random((1, 48, 48, 1)).astype('float32')
            try:
                _ = model.predict(gray_input, verbose=0)
                print(f"Model successfully processes grayscale input with shape {gray_input.shape}")
            except Exception as e:
                print(f"Model FAILS on grayscale input: {e}")
                
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        traceback.print_exc()

def create_simple_emotion_model():
    """Create a simple emotion recognition model with proper input shape"""
    try:
        print("Creating a simple emotion model as fallback")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(emotion_labels), activation='softmax')  
        ])
        
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    except Exception as e:
        print(f"Error creating simple model: {e}")
        traceback.print_exc()
        return None

def preprocess_face_for_emotion(face_image):
    """Apply additional preprocessing to help with emotion recognition based on CNN model"""
    try:
        if model is None:
            print("Warning: Model not loaded in preprocess_face_for_emotion")
            return None
        
        
        expected_shape = get_model_input_shape(model)
        expected_channels = expected_shape[-1] if expected_shape else 1  
        print(f"Model expects {expected_channels} channels")
        
        
        if len(face_image.shape) == 2:
            gray_image = face_image
        else:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        
        resized = cv2.resize(gray_image, (48, 48))
        
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        
        normalized = enhanced.astype("float32") / 255.0
        
        
        if expected_channels == 1:
            normalized = normalized.reshape(48, 48, 1)
        else:
            
            normalized = np.stack([normalized] * 3, axis=-1)
        
        print(f"Preprocessed image shape: {normalized.shape}")
        return normalized
            
    except Exception as e:
        print(f"Error in preprocess_face_for_emotion: {str(e)}")
        traceback.print_exc()
        
        default_img = np.zeros((48, 48, 1), dtype=np.float32)
        return default_img

def balance_emotion_predictions(preds, emotion_labels):
    """Apply bias correction to handle class imbalance"""
    
    bias_correction = {
        'angry': 1.2,
        'disgust': 1.1,  
        'fear': 1.15,    
        'happy': 0.85,   
        'neutral': 1.1,
        'sad': 1.1,
        'surprise': 0.95
    }
    
    adjusted_preds = preds.copy()
    for i, emotion in enumerate(emotion_labels):
        if emotion.lower() in bias_correction:
            adjusted_preds[0][i] *= bias_correction[emotion.lower()]
    
    return adjusted_preds

def detect_faces(image):
    """
    Detect faces in an image using OpenCV's face cascade
    Returns a list of (x, y, w, h) tuples for detected faces
    """
    try:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        
        return faces
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []

def get_model_input_shape(model):
    """Safely get the model's input shape"""
    try:
        
        if hasattr(model, 'input_shape'):
            return model.input_shape
        
        elif hasattr(model, '_feed_input_shapes'):
            return model._feed_input_shapes[0]
        
        else:
            print("Warning: Could not determine model input shape, using default (48,48,1)")
            return (None, 48, 48, 1)
    except Exception as e:
        print(f"Error getting model input shape: {e}")
        return (None, 48, 48, 1)
def detect_emotion(face_img):
    """
    Detect emotion in a preprocessed face image
    Returns (emotion_name, confidence)
    """
    try:
        
        if model is None:
            print("Error: No model loaded for emotion detection")
            return "unknown", 0.0
            
        if face_img is None:
            print("Error: Face image is None")
            return "unknown", 0.0
            
        
        if len(face_img.shape) != 3:
            print(f"Error: Input shape {face_img.shape} is not 3D")
            return "unknown", 0.0
            
        
        face_img_batch = np.expand_dims(face_img, axis=0)
            
        
        expected_shape = model.input_shape
        actual_shape = face_img_batch.shape
        
        print(f"Model expects shape: {expected_shape}, got: {actual_shape}")
        
        
        if expected_shape[-1] != actual_shape[-1]:
            print(f"Warning: Model expects {expected_shape[-1]} channels, but image has {actual_shape[-1]} channels")
            
            
            if expected_shape[-1] == 1 and actual_shape[-1] == 3:
                print("Converting RGB image to grayscale")
                
                face_img_batch = np.mean(face_img_batch, axis=3, keepdims=True)
                print(f"Converted to shape: {face_img_batch.shape}")
            
            
            elif expected_shape[-1] == 3 and actual_shape[-1] == 1:
                print("Converting grayscale image to RGB")
                face_img_batch = np.repeat(face_img_batch, 3, axis=3)
                print(f"Converted to shape: {face_img_batch.shape}")
        
        
        preds = model.predict(face_img_batch, verbose=0)
        
        
        preds = balance_emotion_predictions(preds, emotion_labels)
        
        
        emotion_idx = np.argmax(preds[0])
        
        
        confidence = float(preds[0][emotion_idx])
        
        
        if emotion_idx < len(emotion_labels):
            emotion = emotion_labels[emotion_idx]
        else:
            emotion = "unknown"
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error detecting emotion: {str(e)}")
        traceback.print_exc()
        return "unknown", 0.0

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        
        user_role = request.form.get('userRole', 'general')
        print(f"Processing image upload for user role: {user_role}")
        
        
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
            
        
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join('static', 'uploads', temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        cv2.imwrite(temp_path, img)
        
        
        faces = detect_faces(img)
        
        
        results = []
        result_image = img.copy()
        
        for (x, y, w, h) in faces:
            
            face_roi = img[y:y+h, x:x+w]
            
            
            face_preprocessed = preprocess_face_for_emotion(face_roi)
            
            
            emotion, confidence = detect_emotion(face_preprocessed)
            
            
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })
            
            
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        
        result_filename = f"result_{uuid.uuid4()}.jpg"
        result_path = os.path.join('static', 'results', result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, result_image)
        
        
        user_email = session.get('user', 'anonymous')
        for result in results:
            save_emotion(
                email=user_email,
                emotion=result['emotion'],
                confidence=result['confidence'],
                model_type='image_analysis',
                session_id=None  
            )
        
        
        response = {
            'success': True,
            'facesDetected': len(faces),
            'emotions': results,
            'resultImage': f"/static/results/{result_filename}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




@app.route('/analytics')
def get_analytics():
    try:
        
        time_range = request.args.get('time_range', 'week')
        patient_name = request.args.get('patient_name', None)
        
        print(f"Analytics requested - time_range: {time_range}, patient: {patient_name}")
        
        
        end_time = datetime.now()
        if time_range == 'day':
            start_time = end_time - timedelta(days=1)
        elif time_range == 'week':
            start_time = end_time - timedelta(weeks=1)
        elif time_range == 'month':
            start_time = end_time - timedelta(days=30)
        else:  
            start_time = datetime(2000, 1, 1)  
            
        
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Looking for data between {start_time_str} and now")
        
        
        emotions_data = []
        if os.path.exists(EMOTIONS_CSV):
            with open(EMOTIONS_CSV, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    
                    if row['timestamp'] >= start_time_str:
                        
                        if not patient_name or patient_name == 'all' or row['email'] == patient_name:
                            emotions_data.append({
                                'timestamp': row['timestamp'],
                                'emotion': row['emotion'],
                                'confidence': float(row['confidence']),
                                'model_type': row['model_type'],
                                'session_id': row['session_id'],
                                'patient_name': row['email']
                            })
        
        print(f"Found {len(emotions_data)} emotion records")
        
        
        emotions_by_model = {}
        for entry in emotions_data:
            emotion = entry['emotion']
            if emotion not in emotions_by_model:
                emotions_by_model[emotion] = {
                    'emotion': emotion,
                    'count': 0
                }
            emotions_by_model[emotion]['count'] += 1
        
        
        emotions_by_model_list = list(emotions_by_model.values())
        
        
        emotions_by_time = {}
        for entry in emotions_data:
            try:
                dt = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
                
                minute_key = dt.strftime('%Y-%m-%d %H:%M:00')
                
                
                if minute_key not in emotions_by_time:
                    emotions_by_time[minute_key] = {
                        'timestamp': minute_key,
                        'count': 0,
                        'emotions': {
                            'angry': 0,
                            'disgust': 0,
                            'fear': 0,
                            'happy': 0,
                            'neutral': 0,
                            'sad': 0,
                            'surprise': 0
                        }
                    }
                
                
                emotions_by_time[minute_key]['count'] += 1
                
                
                emotion = entry.get('emotion', 'unknown').lower()
                if emotion in emotions_by_time[minute_key]['emotions']:
                    emotions_by_time[minute_key]['emotions'][emotion] += 1
                
            except Exception as e:
                print(f"Error processing timestamp {entry.get('timestamp')}: {str(e)}")

        
        emotions_by_time_list = list(emotions_by_time.values())
        emotions_by_time_list.sort(key=lambda x: x['timestamp'])

        
        
        session_ids = set([entry['session_id'] for entry in emotions_data if entry['session_id']])
        print(f"Found {len(session_ids)} unique session IDs")
        
        
        session_history = []
        for session_id in session_ids:
            session_emotions = [e for e in emotions_data if e['session_id'] == session_id]
            
            if not session_emotions:
                continue
                
            
            timestamps = [datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S') for e in session_emotions]
            start_time = min(timestamps) if timestamps else None
            end_time = max(timestamps) if timestamps else None
            
            
            if not start_time or not end_time:
                continue
                
            duration = (end_time - start_time).total_seconds() / 60  
            
            
            emotion_counts = {}
            for entry in session_emotions:
                emotion = entry['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            
            
            total = sum(emotion_counts.values())
            emotion_percentages = {
                emotion: (count / total * 100) for emotion, count in emotion_counts.items()
            } if total > 0 else {}
            
            
            model_type = session_emotions[0]['model_type'] if session_emotions else 'Unknown'
            patient_name = session_emotions[0]['patient_name'] if session_emotions else 'Anonymous'
            
            
            session_history.append({
                'sessionId': session_id,
                'startTime': start_time.isoformat(),
                'endTime': end_time.isoformat(),
                'duration': duration,
                'totalDetections': len(session_emotions),
                'emotionBreakdown': emotion_counts,
                'emotionPercentages': emotion_percentages,
                'dominantEmotion': dominant_emotion,
                'modelType': model_type,
                'patientName': patient_name
            })
        
        response_data = {
            'emotionsByModel': emotions_by_model_list,
            'emotionsByTime': emotions_by_time_list,
            'sessionHistory': session_history
        }
        print(f"Returning analytics data with {len(session_history)} sessions")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error getting analytics: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/download-session-data')
def download_session_data():
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
            
        
        session_data = []
        if os.path.exists(EMOTIONS_CSV):
            with open(EMOTIONS_CSV, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['session_id'] == session_id:
                        session_data.append(row)
        
        if not session_data:
            return jsonify({"error": "No data found for this session"}), 404
        
        
        temp_file = f"temp_session_{session_id}.csv"
        with open(temp_file, 'w', newline='') as csvfile:
            fieldnames = ['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in session_data:
                writer.writerow(row)
                
        
        return send_file(
            temp_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"session_{session_id}.csv"
        )
    
    except Exception as e:
        print(f"Error downloading session data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    
    if model is not None:
        inspect_model()
    
    
    
    app.run(debug=True, host='0.0.0.0', port=5005)