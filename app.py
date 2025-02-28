from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model  # type:ignore
import logging
import os
import base64
import time

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Check for required files
required_files = {
    "model.h5": "the trained emotion recognition model",
    "labels.npy": "the emotion labels"
}

missing_files = [f"- {file} ({desc})" for file, desc in required_files.items() if not os.path.exists(file)]
if missing_files:
    print("Required files are missing!\n" + "\n".join(missing_files))
    print("Ensure you've run data_collection.py and data_training.py before using this.")
    exit()

# Load model and labels once
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
    print("✅ Model and labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or labels: {str(e)}")
    exit()

# Initialize MediaPipe Holistic once
holistic = mp.solutions.holistic.Holistic()
hands = mp.solutions.hands

# Frame processing control (skip frames to optimize performance)
last_frame_time = 0
frame_interval = 0.1  # Process frames every 0.1 seconds (10 FPS)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    global last_frame_time

    try:
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            return  # Skip processing if interval not met

        last_frame_time = current_time  # Update last processed frame time

        # Decode the base64 image data
        image_data = data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Process the frame with MediaPipe Holistic
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            # Predict emotion
            pred = label[np.argmax(model.predict(lst))]
            print('🎭 Emotion:', pred)
            socketio.emit('emotion', {'emotion': pred})
        else:
            socketio.emit('emotion', {'error': 'No face detected'})
    
    except Exception as e:
        print(f"⚠️ WebSocket Error: {str(e)}")
        socketio.emit('emotion', {'error': 'Processing error occurred'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
