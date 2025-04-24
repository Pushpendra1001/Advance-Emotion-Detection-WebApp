import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
import os

def create_emotion_model():
    # Create sequential model
    model = Sequential([
        # TimeDistributed layers for video frame processing
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), 
                       input_shape=(1, 96, 96, 3)),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        
        TimeDistributed(Flatten()),
        
        # LSTM layer for temporal features
        LSTM(64, return_sequences=False),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')  # 8 emotions: anger, contempt, disgust, fear, happy, neutral, sad, surprise
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create and save the model
    model = create_emotion_model()
    
    # Print model summary
    model.summary()
    
    # Save model
    model_path = os.path.join(os.getcwd(), 'my_model.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Test model with dummy data
    test_input = np.zeros((1, 1, 96, 96, 3))
    prediction = model.predict(test_input)
    print("\nTest prediction shape:", prediction.shape)
    print("Test prediction:", prediction)

if __name__ == "__main__":
    main()