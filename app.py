from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model  # type:ignore
import logging
import os
import base64
import time
import requests
import random
import json

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# YouTube API configuration
YOUTUBE_API_KEY = "Your-API-key"  # Replace with your actual API key

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

# Emotion to search query mapping
emotion_playlists = {
    "happy": ["happy music", "upbeat songs", "feel good playlist", "joyful tunes", "positive vibes"],
    "sad": ["sad songs", "melancholy music", "heartbreak playlist", "emotional ballads", "sad vibes"],
    "angry": ["angry music", "rage playlist", "heavy metal", "aggressive songs", "fury tracks"],
    "surprise": ["surprising music", "unexpected songs", "musical plot twists", "dramatic scores", "shocking tracks"],
    "fear": ["scary music", "suspense tracks", "horror soundtrack", "tense music", "thriller scores"],
    "disgust": ["intense music", "dark tracks", "unsettling songs", "disturbing scores", "uncomfortable tunes"],
    "neutral": ["relaxing music", "calm playlist", "background tunes", "ambient sounds", "chillout tracks"]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_songs', methods=['POST'])
def get_recommended_songs():
    data = request.get_json()
    emotion = data.get('emotion', 'neutral')
    language = data.get('language', 'English')
    
    # Normalize emotion to match our dictionary keys
    emotion = emotion.lower()
    if emotion not in emotion_playlists:
        emotion = "neutral"
    
    # Get a random search query for the detected emotion
    search_query = random.choice(emotion_playlists[emotion])
    
    # Add language to search query if specified
    if language and language.lower() != "english":
        search_query += f" {language}"
    
    try:
        # Call YouTube API to search for videos
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            'part': 'snippet',
            'q': search_query,
            'type': 'video',
            'videoCategoryId': '10',  # Music category
            'maxResults': 50,  # Get more results to randomize from
            'key': YOUTUBE_API_KEY
        }
        
        response = requests.get(search_url, params=search_params)
        search_data = response.json()
        
        # Randomly select 5 songs from results
        all_songs = search_data.get('items', [])
        recommended_songs = []
        
        if all_songs:
            # Select 5 random songs or all if less than 5
            random_selections = random.sample(all_songs, min(5, len(all_songs)))
            
            for item in random_selections:
                if 'id' in item and 'videoId' in item['id']:
                    song = {
                        'id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'channelTitle': item['snippet']['channelTitle']
                    }
                    recommended_songs.append(song)
        
        return jsonify({
            'emotion': emotion,
            'songs': recommended_songs
        })
    
    except Exception as e:
        print(f"YouTube API Error: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch songs',
            'message': str(e)
        }), 500

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
