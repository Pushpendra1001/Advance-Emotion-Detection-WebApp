import os
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
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_HTTPONLY=True,
    SECRET_KEY='your-secret-key-here'  # Add a secret key for sessions
)

# Simple CORS configuration that should work for development
CORS(app, 
     origins=["http://localhost:5173", "http://127.0.0.1:5173"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

USERS_CSV = 'data/users.csv'
EMOTIONS_CSV = 'data/emotions.csv'

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize CSV files if they don't exist
if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=['email', 'password']).to_csv(USERS_CSV, index=False)
if not os.path.exists(EMOTIONS_CSV):
    pd.DataFrame(columns=['email', 'timestamp', 'emotion', 'confidence', 'model_type', 'session_id']).to_csv(EMOTIONS_CSV, index=False)


model = None
face_cascade = None
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
active_sessions = {}
active_cameras = {}

def load_model_file(model_path):
    global model, face_cascade
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return False
        
        print(f"Loading model from: {model_path}")
        
        # Load the model with error handling - using tf.keras directly
        try:
            # Try different approaches for model loading
            try:
                # First, try standard load_model
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e1:
                print(f"First attempt failed: {str(e1)}")
                # Second, try a custom loading approach
                try:
                    # Try loading with skip_mismatch option (for incompatible architectures)
                    model = tf.keras.models.load_model(
                        model_path, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )
                except Exception as e2:
                    print(f"Second attempt failed: {str(e2)}")
                    # Last resort: Create a simple model with similar structure
                    print("Attempting to rebuild model and load weights only")
                    # Build a basic CNN model
                    inputs = tf.keras.Input(shape=(96, 96, 3))
                    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Flatten()(x)
                    x = tf.keras.layers.Dense(128, activation='relu')(x)
                    outputs = tf.keras.layers.Dense(8, activation='softmax')(x)
                    
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    try:
                        model.load_weights(model_path)
                    except:
                        return False
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading the Keras model: {str(e)}")
            return False
        
        # Load face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("Error: Could not load face cascade classifier")
                return False
            print("Face cascade loaded successfully")
        except Exception as e:
            print(f"Error loading face cascade: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Unexpected error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def validate_model_path(model_path):
    """Validate and fix model path if necessary"""
    # If path is already absolute, use it
    if os.path.isabs(model_path):
        if os.path.exists(model_path):
            return model_path
        else:
            return None
    
    # Try different locations based on relative path
    potential_paths = [
        model_path,  # As provided
        os.path.join(os.getcwd(), model_path),  # Relative to current working directory
        os.path.join(os.getcwd(), 'models', model_path),  # In models subdirectory
        os.path.join(os.getcwd(), 'frontend', model_path),  # In frontend subdirectory
        os.path.join(os.getcwd(), 'frontend', '/models/emotion_model.keras'),  # Fixed path in frontend
        os.path.join(os.getcwd(), 'backend', 'models', '/models/emotion_model.keras')  # Path in backend/models
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    # If all else fails, see if we can find any .keras files
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.keras'):
                full_path = os.path.join(root, file)
                print(f"Found potential model file: {full_path}")
                return full_path
    
    return None
    
def save_emotion(email, emotion, confidence, model_type, session_id=None):
    try:
        timestamp = datetime.now()
        new_data = {
            'email': email,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'emotion': emotion,
            'confidence': confidence,
            'model_type': model_type,
            'session_id': session_id
        }
        
        # Add to active session if session_id provided
        if session_id and session_id in active_sessions:
            active_sessions[session_id]['emotions'].append({
                **new_data,
                'timestamp': timestamp
            })
        
        # Save to CSV
        try:
            df = pd.read_csv(EMOTIONS_CSV)
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(EMOTIONS_CSV, index=False)
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            
    except Exception as e:
        print(f"Error saving emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        
        
def process_frame(frame, email, model_type, session_id):
    try:
        if model is None:
            print("No model loaded. Please load a model first.")
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (96, 96))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            
            # Check model input shape requirements
            input_shape = model.input_shape
            if len(input_shape) > 3:  # Model expects batch dimension
                if input_shape[1:] != (96, 96, 3):
                    print(f"Warning: Model expects input shape {input_shape} but got (96, 96, 3)")
            else:  # Model doesn't have batch dimension in shape
                if input_shape != (96, 96, 3):
                    print(f"Warning: Model expects input shape {input_shape} but got (96, 96, 3)")
            
            prediction = model.predict(face)
            emotion_idx = np.argmax(prediction)
            
            # Ensure emotion_idx is within range
            if emotion_idx >= len(emotion_labels):
                emotion_idx = 0
                print(f"Warning: Prediction index {emotion_idx} out of range. Using default.")
                
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[0][emotion_idx])
            
            save_emotion(email, emotion, confidence, model_type, session_id)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error in processing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        # Draw error message on frame
        cv2.putText(frame, "Error processing frame", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

@app.route('/check-models')
def check_models():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Look for models in multiple directories
    model_files = []
    
    # Check in current directory and subdirectories
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
    
    while session_id in active_sessions:  # Check if session is still active
        success, frame = cap.read()
        if not success:
            break
        
        try:
            processed_frame = process_frame(frame, email, model_type, session_id)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
    
    # Clean up camera when done
    if cap and cap.isOpened():
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
        return jsonify({"status": "success", "email": email})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        email = session['user']
        
        # Check if file exists and is not empty
        if not os.path.exists(EMOTIONS_CSV) or os.stat(EMOTIONS_CSV).st_size == 0:
            return jsonify({
                "sessionHistory": [],
                "emotionsByTime": [],
                "emotionsByModel": [],
                "emotionTrends": []
            })
            
        df = pd.read_csv(EMOTIONS_CSV)
        user_data = df[df['email'] == email]
        
        if user_data.empty:
            return jsonify({
                "sessionHistory": [],
                "emotionsByTime": [],
                "emotionsByModel": [],
                "emotionTrends": []
            })

        # Process each session separately
        session_summaries = []
        for session_id in user_data['session_id'].unique():
            session_data = user_data[user_data['session_id'] == session_id]
            
            # Calculate emotion counts
            emotion_counts = session_data['emotion'].value_counts().to_dict()
            total_emotions = len(session_data)
            
            # Calculate percentages
            emotion_percentages = {
                emotion: round((count / total_emotions * 100), 2)
                for emotion, count in emotion_counts.items()
            }
            
            session_summaries.append({
                'sessionId': session_id,
                'startTime': session_data['timestamp'].min(),
                'endTime': session_data['timestamp'].max(),
                'duration': round((pd.to_datetime(session_data['timestamp'].max()) - 
                                pd.to_datetime(session_data['timestamp'].min())).total_seconds() / 60, 2),
                'modelType': session_data['model_type'].iloc[0],
                'dominantEmotion': max(emotion_counts.items(), key=lambda x: x[1])[0],
                'totalDetections': total_emotions,
                'emotionBreakdown': emotion_counts,
                'emotionPercentages': emotion_percentages
            })
        
        # Sort sessions by start time, most recent first
        session_summaries.sort(key=lambda x: x['startTime'], reverse=True)
        
        return jsonify({
            "sessionHistory": session_summaries,
            "emotionsByTime": user_data.groupby(['timestamp', 'emotion'])
                .size()
                .reset_index(name='count')
                .to_dict('records'),
            "emotionsByModel": user_data.groupby(['model_type', 'emotion'])
                .size()
                .reset_index(name='count')
                .to_dict('records'),
            "emotionTrends": []  # Add empty trends array as default
        })
        
    except Exception as e:
        print(f"Analytics Error: {str(e)}")
        import traceback
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

# Update the model_status endpoint
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
        
        # Validate and fix model path
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
        # Get parameters from query string
        model_type = request.args.get('model_type', 'general-analysis')
        session_id = request.args.get('session_id')
        email = session.get('user', 'test@example.com')  # Get email from session or use default
        
        if not model_type:
            return jsonify({"error": "Model type is required"}), 400
             
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "No model loaded"}), 400
             
        # Create and return the video feed response
        response = Response(
            generate_frames(email, model_type, session_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Add this helper function to create a preview frame with text
def create_preview_frame(text):
    # Create a black image
    frame = np.zeros((480, 640, 3), np.uint8)
    
    # Add text to the image
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

@app.route('/start-session', methods=['POST'])
def start_session():
    # For local testing, comment this out
    # if not session.get('user'):
    #     return jsonify({"error": "Unauthorized"}), 401
    
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "No model loaded. Please load a model first."}), 400
    
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "user": session.get("user", "test@example.com"),  # Provide a default email for testing
        "start_time": datetime.now(),
        "emotions": []
    }
    
    return jsonify({"sessionId": session_id})

@app.route('/stop-session', methods=['POST'])
def stop_session():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    session_id = data.get("sessionId")

    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Invalid session"}), 400

    session_data = active_sessions[session_id]
    duration = (datetime.now() - session_data['start_time']).total_seconds() / 60

    # Calculate session statistics
    emotions = [e['emotion'] for e in session_data['emotions']]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None

    # Calculate percentages for each emotion
    total_emotions = len(emotions)
    emotion_percentages = {
        emotion: (count / total_emotions * 100) if total_emotions > 0 else 0 
        for emotion, count in emotion_counts.items()
    }

    report = {
        "duration": round(duration, 2),
        "dominantEmotion": dominant_emotion,
        "totalDetections": total_emotions,
        "emotionBreakdown": emotion_counts,
        "emotionPercentages": {
            emotion: round(percentage, 2) 
            for emotion, percentage in emotion_percentages.items()
        }
    }

    # Clean up camera if it exists
    if session_id in active_cameras:
        cap = active_cameras[session_id]
        if cap and cap.isOpened():
            cap.release()
        del active_cameras[session_id]

    # Clean up session
    del active_sessions[session_id]

    return jsonify(report)

@app.after_request
def after_request(response):
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Max-Age', '3600')
    # For regular requests
    else:
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    return response

@app.route('/status')
def status():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({"status": "running"})

if __name__ == '__main__':
    # Attempt to load a default model if available
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