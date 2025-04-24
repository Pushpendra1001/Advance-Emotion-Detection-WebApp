import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import cv2
import os

def create_emotion_model():
    model = Sequential([
        # First Convolutional Block - Note the input_shape is (96, 96, 3)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96, 96, 3)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')  # 8 emotions
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_model(model, face_cascade):
    """Test the model with a sample image from webcam"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
            
        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame")
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model - note we're not adding the extra time dimension
            face_roi = cv2.resize(face_roi, (96, 96))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)  # Add only batch dimension
            
            # Make prediction
            prediction = model.predict(face_roi, verbose=0)
            
            # Print results with emotion labels
            emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            for emotion, prob in zip(emotion_labels, prediction[0]):
                print(f"{emotion}: {prob*100:.2f}%")
            
            return True
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()

def main():
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error loading face cascade classifier")
    
    # Create model
    print("Creating model...")
    model = create_emotion_model()
    
    # Print model summary
    model.summary()
    
    # Test the model
    print("\nTesting model with webcam...")
    test_success = test_model(model, face_cascade)
    
    if test_success:
        # Save model
        model_path = os.path.join(os.getcwd(), 'my_model.keras')
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    else:
        print("\nModel test failed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")