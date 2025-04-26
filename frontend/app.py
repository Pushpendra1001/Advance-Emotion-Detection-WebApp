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
# Use a simplified set of emotion labels that match your custom model
emotion_labels = ['angry', 'happy', 'neutral']  # Only 3 emotions in your custom model
active_sessions = {}
active_cameras = {}

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
    # Convert to RGB (ensure consistent color space)
    if len(face_image.shape) == 2:  # If grayscale
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.shape[2] == 3:  # If BGR (OpenCV default)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Resize to the target size (48x48) which is common for emotion models
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
    
    # Print shapes for debugging
    print(f"Preprocessed image shape: {normalized.shape}")
    
    # Don't add channel dimension - ensure it remains (48, 48, 3)
    return normalized

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

if __name__ == '__main__':
    # First inspect the model to understand what it expects
    if model is not None:
        inspect_model()
    
    # Update the model validation part in the main execution block

    if model is not None:
        # Get the expected input shape
        expected_input_shape = model.input_shape
        print(f"Model expects input shape: {expected_input_shape}")
        
        # Test if the model can process a simple input with the right shape
        try:
            # Create a test input with the right channel dimension
            if expected_input_shape[3] == 3:  # RGB
                test_input = np.random.random((1, 48, 48, 3)).astype('float32')
                print("Testing with RGB input")
            else:  # Grayscale
                test_input = np.random.random((1, 48, 48, 1)).astype('float32')
                print("Testing with grayscale input")
                
            predictions = model.predict(test_input, verbose=0)
            
            # Verify the output shape matches our emotion labels
            if predictions.shape[1] == len(emotion_labels):
                model_valid = True
                print("Model validated successfully with test input")
                print(f"Model predicts {len(emotion_labels)} emotions: {emotion_labels}")
            else:
                print(f"Model output shape {predictions.shape} doesn't match our {len(emotion_labels)} emotion labels")
                # Try to create a new compatible model if shapes don't match
                model = create_simple_emotion_model()
                if model:
                    model_valid = True
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

    app.run(debug=True, host='0.0.0.0', port=5005)