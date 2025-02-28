import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model  # type: ignore
import webbrowser
import os
import logging

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Check for required files
required_files = {
    "model.h5": "the trained emotion recognition model",
    "labels.npy": "the emotion labels",
    "emotion.npy": "the detected emotion"
}

missing_files = [f"- {file} ({desc})" for file, desc in required_files.items() if not os.path.exists(file)]
if missing_files:
    print("Required files are missing!\n" + "\n".join(missing_files))
    print("Ensure you've run data_collection.py and data_training.py before using this.")
    exit()

# Load model and labels
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
    print("Model and labels loaded successfully!")
except Exception as e:
    print(f"Error loading model or labels: {str(e)}")
    exit()

# Initialize MediaPipe Holistic
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

# Load previously detected emotion if available
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

# Capture video
cap = cv2.VideoCapture(0)

print("Enter preferred language: ", end="")
lang = input().strip()
print("Enter preferred singer (optional): ", end="")
singer = input().strip()

if not lang:
    print("Language is required for recommendations.")
    cap.release()
    exit()

print("Look at the camera to detect your emotion...")

with holistic.Holistic() as holis:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        res = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

            try:
                pred = label[np.argmax(model.predict(lst))]
                cv2.putText(frame, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
                emotion = pred
            except Exception as e:
                print(f"Prediction error: {str(e)}")

        # Draw landmarks
        if res.face_landmarks:
            drawing.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        if res.left_hand_landmarks:
            drawing.draw_landmarks(frame, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        if res.right_hand_landmarks:
            drawing.draw_landmarks(frame, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Recommend songs based on detected emotion
if not emotion:
    print("No emotion detected. Please try again.")
else:
    search_query = f"{lang}+{emotion}+song"
    if singer:
        search_query += f"+{singer}"
    webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
    np.save("emotion.npy", np.array([""]))
