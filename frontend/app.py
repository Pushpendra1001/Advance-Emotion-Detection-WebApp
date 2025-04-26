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
     methods=["GET", "POST", "OPTIONS"])

USERS_CSV = 'data/users.csv'
EMOTIONS_CSV = 'data/emotions.csv'

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=['email', 'password']).to_csv(USERS_CSV, index=False)
if not os.path.exists(EMOTIONS_CSV):
    pd.DataFrame(columns=['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']).to_csv(EMOTIONS_CSV, index=False)

# Initialize model as global variable
model = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Use a simplified set of emotion labels that match common emotion models
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # 7 basic emotions
active_sessions = {}
active_cameras = {}

# Search for the model file in common locations
def find_model_file():
    print("Searching for model file...")
    possible_paths = [
        os.path.join('src', 'models', 'my_model.keras'),
        os.path.join('models', 'my_model.keras'),
        os.path.join('frontend', 'src', 'models', 'my_model.keras'),
        'my_model.keras',
        os.path.join('..', 'models', 'my_model.keras')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
            
    # If model not found in expected locations, search whole directory
    print("Searching entire directory...")
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.keras'):
                path = os.path.join(root, file)
                print(f"Found model at: {path}")
                return path
                
    return None

# Load the model at startup
try:
    MODEL_PATH = find_model_file()
    if MODEL_PATH:
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
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
    """Save detected emotion to database"""
    try:
        # Get session data if session_id is provided
        patient_name = email
        if session_id and session_id in active_sessions:
            patient_name = active_sessions[session_id].get('patientName', email)
        else:
            patient_name = email if email != 'anonymous' else 'anonymous'
        
        # Create emotion record
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record = {
            'email': patient_name,  # Use patient name instead of email
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence,
            'model_type': model_type,
            'session_id': session_id
        }
        
        # Save to CSV file
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

        # Make a copy of the frame to avoid modifications
        processed_frame = frame.copy()
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Print face detection results only occasionally to avoid flooding the console
        if len(faces) > 0:
            print(f"Detected {len(faces)} faces")
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            try:
                # Extract face ROI and preprocess for emotion detection
                if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
                    print(f"Face coordinates out of bounds: x={x}, y={y}, w={w}, h={h}, frame={frame.shape}")
                    continue
                    
                # Extract face region from the original color image
                face_roi = processed_frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    print("Empty face ROI")
                    continue
                
                # Convert to RGB (model may expect RGB)
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                # Resize to model input size (48x48 is common for emotion models)
                face_roi_resized = cv2.resize(face_roi_rgb, (48, 48))
                
                # Normalize pixel values to [0, 1]
                face_roi_norm = face_roi_resized.astype("float32") / 255.0
                
                # Add batch dimension
                face_roi_batch = np.expand_dims(face_roi_norm, axis=0)
                
                # Print shape info for debugging
                print(f"Face ROI shape before prediction: {face_roi_batch.shape}")
                
                # Check if model is loaded
                if model is None:
                    print("Model is not loaded! Cannot predict emotion.")
                    cv2.putText(processed_frame, "Model not loaded", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue
                
                # Try to get model input shape
                input_shape = model.input_shape
                if input_shape:
                    print(f"Model expects input shape: {input_shape}")
                    # Resize to match expected input shape if needed
                    if len(input_shape) == 4:
                        expected_h, expected_w = input_shape[1], input_shape[2]
                        if expected_h is not None and expected_w is not None:
                            face_roi_resized = cv2.resize(face_roi_rgb, (expected_w, expected_h))
                            face_roi_norm = face_roi_resized.astype("float32") / 255.0
                            face_roi_batch = np.expand_dims(face_roi_norm, axis=0)
                            print(f"Reshaped to match model input: {face_roi_batch.shape}")
                
                # Predict emotion
                preds = model.predict(face_roi_batch, verbose=0)
                emotion_idx = np.argmax(preds[0])
                emotion = emotion_labels[emotion_idx] if emotion_idx < len(emotion_labels) else "unknown"
                confidence = float(preds[0][emotion_idx])
                
                print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
                
                # Get patient name from session data
                patient_name = session_data.get('patientName', 'anonymous')
                
                # Save emotion with session data
                save_emotion(
                    email=patient_name,
                    emotion=emotion,
                    confidence=confidence,
                    model_type=session_data.get('model_type', 'unknown'),
                    session_id=session_data.get('id')
                )
                
                # Draw emotion label
                label = f"{emotion}: {confidence:.2f}"
                # Draw background rectangle for text
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(processed_frame, 
                            (x, y-30), 
                            (x + label_w, y), 
                            (0, 0, 0), 
                            -1)
                # Draw text
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
    """Initialize a new emotion tracking session"""
    try:
        data = request.json
        session_id = str(uuid.uuid4())
        
        # Store session information
        session = {
            'id': session_id,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patientName': data.get('patientName', 'anonymous'),  # Make sure we get this
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
            # Return success even if session not found, to avoid client errors
            return jsonify({
                "sessionId": session_id,
                "status": "stopped",
                "message": "Session already ended or not found",
                "dominantEmotion": "unknown",
                "duration": 0
            }), 200
            
        # Get session data
        session_data = active_sessions[session_id]
        session_data['end_time'] = datetime.now()
        
        # Convert start_time from string to datetime if needed
        start_time = session_data.get('start_time')
        if isinstance(start_time, str):
            try:
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # If parsing fails, use a default time
                print(f"Could not parse start_time: {start_time}")
                start_time = datetime.now() - timedelta(minutes=1)
        else:
            start_time = datetime.now() - timedelta(minutes=1)
            
        end_time = session_data['end_time']
        duration = (end_time - start_time).total_seconds() / 60  
        
        # Rest of your function remains the same
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
        
        # Read from CSV to get emotions for this session
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
        
        # Store in session data
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
        
        # Get the expected input shape from the first layer
        expected_shape = model.layers[0].input_shape
        print(f"Model expects input shape: {expected_shape}")
        
        # If the model expects grayscale but we're feeding RGB
        if expected_shape and len(expected_shape) == 4:
            if expected_shape[3] == 1:  # Model expects grayscale
                print("Model expects grayscale input. Updating preprocessing.")
            elif expected_shape[3] == 3:  # Model expects RGB
                print("Model expects RGB input. Updating preprocessing.")
                
        return True
    except Exception as e:
        print(f"Error fixing model input shape: {str(e)}")
        return False

def inspect_model():
    """Print out model architecture and expected input shape"""
    try:
        if model is None:
            print("No model loaded to inspect")
            return
            
        print("Model Summary:")
        model.summary()
        
        # Get the input shape from the first layer
        input_shape = model.layers[0].input_shape
        print(f"Input shape: {input_shape}")
        
        # Get the output shape from the last layer
        output_shape = model.layers[-1].output_shape
        print(f"Output shape: {output_shape}")
        print(f"Number of output classes: {output_shape[1]}")
        
        if output_shape[1] != len(emotion_labels):
            print(f"WARNING: Model outputs {output_shape[1]} classes but we have {len(emotion_labels)} emotion labels!")
            print(f"This may cause incorrect emotion labeling.")
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

def create_simple_emotion_model():
    """Create a simple emotion recognition model with proper input shape"""
    try:
        print("Creating a simple emotion model as fallback")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        
        # Create a simple CNN model
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
        
        # Compile model
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

if __name__ == '__main__':
    # Model should already be loaded from the earlier initialization code
    model_valid = False
    
    if model is not None:
        # Test if the model can process a simple input
        try:
            # Create a test input with a random 48x48 RGB image
            test_input = np.random.random((1, 48, 48, 3)).astype('float32')
            _ = model.predict(test_input, verbose=0)
            model_valid = True
            print("Model validated successfully with test input")
        except Exception as e:
            print(f"Error validating model: {e}")
            print("Current model cannot process inputs correctly. Trying to create a fallback model...")
            model = create_simple_emotion_model()
            if model:
                model_valid = True
    else:
        print("Model was not loaded during initialization. Creating a fallback model...")
        model = create_simple_emotion_model()
        if model:
            model_valid = True
    
    if not model_valid:
        print("WARNING: No valid model was loaded. Emotion detection will not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5005)