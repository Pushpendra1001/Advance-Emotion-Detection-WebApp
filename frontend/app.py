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
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  # Add all methods
     allow_headers=["Content-Type", "Authorization"])      # Add allowed headers

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
# Use a simplified set of emotion labels that match your custom model
emotion_labels = ['angry', 'happy', 'neutral']  # Only 3 emotions in your custom model
active_sessions = {}
active_cameras = {}

# Add this after where you create the data directory
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Search for the model file in common locations
def find_model_file():
    print("Searching for model file...")
    possible_paths = [
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
            
    # If model not found in expected locations, search whole directory
    print("Searching entire directory for custom emotion model...")
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if 'emotion_recognition_model_filtered' in file and file.endswith('.keras'):
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
                
                # Resize to 48x48 (common size for emotion models)
                face_roi_resized = cv2.resize(face_roi_rgb, (48, 48))
                
                # Normalize pixel values to [0, 1]
                face_roi_norm = preprocess_face_for_emotion(face_roi_rgb)

                # Add batch dimension
                face_roi_batch = np.expand_dims(face_roi_norm, axis=0)
                
                # Check if model is loaded
                if model is None:
                    print("Model is not loaded! Cannot predict emotion.")
                    cv2.putText(processed_frame, "Model not loaded", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue
                
                # Check model's input shape
                model_input_shape = model.input_shape
                print(f"Model input shape: {model_input_shape}")

                # Make sure face_roi_batch has the correct shape
                if model_input_shape[3] == 3 and face_roi_batch.shape[3] != 3:
                    # Convert grayscale to RGB
                    grayscale = np.squeeze(face_roi_batch, axis=3)
                    face_roi_batch = np.stack([grayscale, grayscale, grayscale], axis=-1)
                    print(f"Converted grayscale to RGB: {face_roi_batch.shape}")
                elif model_input_shape[3] == 1 and face_roi_batch.shape[3] != 1:
                    # Convert RGB to grayscale
                    face_roi_batch = np.mean(face_roi_batch, axis=3, keepdims=True)
                    print(f"Converted RGB to grayscale: {face_roi_batch.shape}")

                # Predict emotion with your custom model
                preds = model.predict(face_roi_batch, verbose=0)

                # Apply bias correction for class imbalance
                preds = balance_emotion_predictions(preds, emotion_labels)
                emotion_idx = np.argmax(preds[0])
                
                # Apply confidence threshold - ignore low confidence predictions
                MIN_CONFIDENCE = 0.40  # Adjust this threshold as needed
                if np.max(preds[0]) < MIN_CONFIDENCE:
                    print(f"Low confidence prediction: {np.max(preds[0]):.2f} - treating as neutral")
                    # Find the index for neutral in your emotion_labels
                    neutral_idx = emotion_labels.index('neutral') if 'neutral' in emotion_labels else None
                    if neutral_idx is not None:
                        emotion_idx = neutral_idx
                        confidence = float(preds[0][emotion_idx])
                    else:
                        # If no neutral class, use the highest confidence one but note it
                        emotion = emotion_labels[emotion_idx]
                        confidence = float(preds[0][emotion_idx])
                        print(f"Using low confidence emotion: {emotion}")
                else:
                    # Normal case - good confidence
                    confidence = float(preds[0][emotion_idx])

                # Map prediction index to emotion label
                if emotion_idx < len(emotion_labels):
                    emotion = emotion_labels[emotion_idx]
                else:
                    print(f"Warning: Model predicted class {emotion_idx} but we only have {len(emotion_labels)} labels")
                    emotion = "unknown"
                    confidence = 0.0

                # Print all emotion scores for debugging
                emotion_scores = {emotion_labels[i]: float(preds[0][i]) for i in range(len(emotion_labels))}
                print(f"All emotion scores: {emotion_scores}")
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
        
        # Add model inspection to help debug
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
    """Print out detailed model architecture and expected input shape"""
    try:
        if model is None:
            print("No model loaded to inspect")
            return
            
        # Print model summary
        print("===== MODEL DETAILS =====")
        model.summary()
        
        # Get the input information
        input_shape = model.input_shape
        print(f"Model input shape: {input_shape}")
        
        # Check the first layer specifically
        first_layer = model.layers[0]
        print(f"First layer: {first_layer.name}, Input shape: {first_layer.input_shape}")
        
        # Get the output shape
        output_shape = model.layers[-1].output_shape
        print(f"Output shape: {output_shape}")
        print(f"Number of output classes: {output_shape[1]}")
        
        # Compare output classes with emotion labels
        if output_shape[1] != len(emotion_labels):
            print(f"WARNING: Model outputs {output_shape[1]} classes but we have {len(emotion_labels)} emotion labels!")
            print("This mismatch could cause incorrect emotion labeling.")
        
        # Create a test input to see if the model accepts it
        if input_shape[1:]:
            # RGB test input
            rgb_input = np.random.random((1, 48, 48, 3)).astype('float32')
            try:
                _ = model.predict(rgb_input, verbose=0)
                print(f"Model successfully processes RGB input with shape {rgb_input.shape}")
            except Exception as e:
                print(f"Model FAILS on RGB input: {e}")
            
            # Grayscale test input
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
        
        # Create a simple CNN model that only predicts the 3 emotions we care about
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
            Dense(len(emotion_labels), activation='softmax')  # Only 3 outputs for angry, happy, neutral
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

def preprocess_face_for_emotion(face_image):
    """Apply additional preprocessing to help with emotion recognition"""
    try:
        # First check if model is loaded
        if model is None:
            print("Warning: Model not loaded in preprocess_face_for_emotion")
            return None
            
        # Get the expected input shape from model
        # Fix: Use model.input_shape instead of model.layers[0].input_shape
        expected_shape = model.input_shape
        expected_channels = expected_shape[-1] if expected_shape else 3
        print(f"Model expects {expected_channels} channels")
        
        # Convert to grayscale if model expects 1 channel
        if expected_channels == 1:
            # If image is already grayscale, just ensure it's the right format
            if len(face_image.shape) == 2:
                gray_image = face_image
            else:
                # Convert from BGR/RGB to grayscale
                gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                
            # Resize to the target size (48x48)
            resized = cv2.resize(gray_image, (48, 48))
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            
            # Normalize pixel values
            normalized = enhanced.astype("float32") / 255.0
            
            # Reshape to include channel dimension (48, 48) -> (48, 48, 1)
            normalized = normalized.reshape(48, 48, 1)
            
            print(f"Preprocessed grayscale image shape: {normalized.shape}")
            return normalized
            
        else:  # RGB processing (3 channels)
            # Ensure consistent color space (convert to RGB if needed)
            if len(face_image.shape) == 2:  # If grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 3:  # If BGR (OpenCV default)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to the target size (48x48)
            resized = cv2.resize(face_image, (48, 48))
            
            # Apply contrast enhancement
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Normalize pixel values
            normalized = enhanced_rgb.astype("float32") / 255.0
            
            print(f"Preprocessed RGB image shape: {normalized.shape}")
            return normalized
            
    except Exception as e:
        print(f"Error in preprocess_face_for_emotion: {str(e)}")
        traceback.print_exc()
        # Return a default normalized grayscale image in case of error
        default_img = np.zeros((48, 48, 1), dtype=np.float32)
        return default_img

def balance_emotion_predictions(preds, emotion_labels):
    """Apply bias correction to handle class imbalance"""
    # Adjusts prediction scores to compensate for dataset bias
    # These values should be tuned based on your model's behavior
    bias_correction = {
        'angry': 1.2,   # Boost angry predictions slightly
        'happy': 0.85,  # Reduce happy predictions as it's overrepresented
        'neutral': 1.1  # Slightly boost neutral
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
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use the face cascade to detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Return the list of face coordinates
        return faces
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []
# Add this helper function somewhere in your code
def get_model_input_shape(model):
    """Safely get the model's input shape"""
    try:
        # First try direct attribute
        if hasattr(model, 'input_shape'):
            return model.input_shape
        # Then try getting from inputs
        elif hasattr(model, '_feed_input_shapes'):
            return model._feed_input_shapes[0]
        # Last resort - use a standard shape
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
        # Check if model is loaded
        if model is None:
            print("Error: No model loaded for emotion detection")
            return "unknown", 0.0
            
        if face_img is None:
            print("Error: Face image is None")
            return "unknown", 0.0
            
        # Check if the input has the right shape
        if len(face_img.shape) != 3:
            print(f"Error: Input shape {face_img.shape} is not 3D")
            return "unknown", 0.0
            
        # Add batch dimension if needed
        face_img_batch = np.expand_dims(face_img, axis=0)
            
        # Fix: Use model.input_shape instead of model.layers[0].input_shape
        expected_shape = model.input_shape
        actual_shape = face_img_batch.shape
        
        print(f"Model expects shape: {expected_shape}, got: {actual_shape}")
        
        # Make sure we match the expected channel count
        if expected_shape[-1] != actual_shape[-1]:
            print(f"Warning: Model expects {expected_shape[-1]} channels, but image has {actual_shape[-1]} channels")
            
            # Convert RGB to grayscale if needed
            if expected_shape[-1] == 1 and actual_shape[-1] == 3:
                print("Converting RGB image to grayscale")
                # Extract first channel or compute average
                face_img_batch = np.mean(face_img_batch, axis=3, keepdims=True)
                print(f"Converted to shape: {face_img_batch.shape}")
            
            # Convert grayscale to RGB if needed
            elif expected_shape[-1] == 3 and actual_shape[-1] == 1:
                print("Converting grayscale image to RGB")
                face_img_batch = np.repeat(face_img_batch, 3, axis=3)
                print(f"Converted to shape: {face_img_batch.shape}")
        
        # Get predictions
        preds = model.predict(face_img_batch, verbose=0)
        
        # Apply bias correction
        preds = balance_emotion_predictions(preds, emotion_labels)
        
        # Get the index of the highest prediction
        emotion_idx = np.argmax(preds[0])
        
        # Get the confidence value
        confidence = float(preds[0][emotion_idx])
        
        # Get the emotion label
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
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Get user role from form data
        user_role = request.form.get('userRole', 'general')
        print(f"Processing image upload for user role: {user_role}")
        
        # Read and decode the image
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
            
        # Save original image to a temporary file
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join('static', 'uploads', temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        cv2.imwrite(temp_path, img)
        
        # Detect faces
        faces = detect_faces(img)
        
        # Process each face for emotion detection
        results = []
        result_image = img.copy()
        
        for (x, y, w, h) in faces:
            # Extract face region from the original color image
            face_roi = img[y:y+h, x:x+w]
            
            # Preprocess face for emotion detection
            face_preprocessed = preprocess_face_for_emotion(face_roi)
            
            # Detect emotion
            emotion, confidence = detect_emotion(face_preprocessed)
            
            # Add result
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })
            
            # Draw rectangle and label on result image
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save result image
        result_filename = f"result_{uuid.uuid4()}.jpg"
        result_path = os.path.join('static', 'results', result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, result_image)
        
        # Save results for anonymous or authenticated user
        user_email = session.get('user', 'anonymous')
        for result in results:
            save_emotion(
                email=user_email,
                emotion=result['emotion'],
                confidence=result['confidence'],
                model_type='image_analysis',
                session_id=None  # No active session for image uploads
            )
        
        # Return results
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

# Move and restructure - delete the current analytics endpoints at the end of the file
# and place this version before if __name__ == '__main__':

@app.route('/analytics')
def get_analytics():
    try:
        # Get query parameters
        time_range = request.args.get('time_range', 'week')
        patient_name = request.args.get('patient_name', None)
        
        print(f"Analytics requested - time_range: {time_range}, patient: {patient_name}")
        
        # Calculate date range based on time_range
        end_time = datetime.now()
        if time_range == 'day':
            start_time = end_time - timedelta(days=1)
        elif time_range == 'week':
            start_time = end_time - timedelta(weeks=1)
        elif time_range == 'month':
            start_time = end_time - timedelta(days=30)
        else:  # 'all'
            start_time = datetime(2000, 1, 1)  # Long time ago
            
        # Format as string for comparison
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Looking for data between {start_time_str} and now")
        
        # Read emotion data from CSV
        emotions_data = []
        if os.path.exists(EMOTIONS_CSV):
            with open(EMOTIONS_CSV, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Apply time range filter
                    if row['timestamp'] >= start_time_str:
                        # Apply patient filter if specified
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
        
        # Process emotions by model type
        emotions_by_model = {}
        for entry in emotions_data:
            emotion = entry['emotion']
            if emotion not in emotions_by_model:
                emotions_by_model[emotion] = {
                    'emotion': emotion,
                    'count': 0
                }
            emotions_by_model[emotion]['count'] += 1
        
        # Convert to list
        emotions_by_model_list = list(emotions_by_model.values())
        
        # Process emotions by time
        emotions_by_time = []
        for entry in emotions_data:
            emotions_by_time.append({
                'timestamp': entry['timestamp'],
                'emotion': entry['emotion'],
                'count': 1
            })
        
        # Extract unique session IDs
        session_ids = set([entry['session_id'] for entry in emotions_data if entry['session_id']])
        print(f"Found {len(session_ids)} unique session IDs")
        
        # Build session history
        session_history = []
        for session_id in session_ids:
            session_emotions = [e for e in emotions_data if e['session_id'] == session_id]
            
            if not session_emotions:
                continue
                
            # Extract timestamps to calculate duration
            timestamps = [datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S') for e in session_emotions]
            start_time = min(timestamps) if timestamps else None
            end_time = max(timestamps) if timestamps else None
            
            # Skip if no valid timestamps
            if not start_time or not end_time:
                continue
                
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Count emotions in this session
            emotion_counts = {}
            for entry in session_emotions:
                emotion = entry['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Calculate dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            
            # Calculate percentages
            total = sum(emotion_counts.values())
            emotion_percentages = {
                emotion: (count / total * 100) for emotion, count in emotion_counts.items()
            } if total > 0 else {}
            
            # Get model type and patient name
            model_type = session_emotions[0]['model_type'] if session_emotions else 'Unknown'
            patient_name = session_emotions[0]['patient_name'] if session_emotions else 'Anonymous'
            
            # Add to session history
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
            'emotionsByTime': emotions_by_time,
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
            
        # Read data from CSV
        session_data = []
        if os.path.exists(EMOTIONS_CSV):
            with open(EMOTIONS_CSV, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['session_id'] == session_id:
                        session_data.append(row)
        
        if not session_data:
            return jsonify({"error": "No data found for this session"}), 404
        
        # Create a temporary CSV file
        temp_file = f"temp_session_{session_id}.csv"
        with open(temp_file, 'w', newline='') as csvfile:
            fieldnames = ['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in session_data:
                writer.writerow(row)
                
        # Send the file
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

# Move this section before app.run()

if __name__ == '__main__':
    # First inspect the model to understand what it expects
    if model is not None:
        inspect_model()
    
    # ... rest of your model validation code ...
    
    app.run(debug=True, host='0.0.0.0', port=5005)