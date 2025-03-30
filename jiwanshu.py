from gtts import gTTS
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
import queue

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 100)  # Speed of speech

# Speech Queue
sentence_queue = queue.Queue()

# Function to Speak a Full Sentence
def speech_worker():
    while True:
        sentence = sentence_queue.get()  # Wait for a sentence to be added to the queue
        if sentence == "EXIT":
            break
        print("Speaking:", sentence)  # Debugging
        engine.say(sentence)
        engine.runAndWait()
        sentence_queue.task_done()

# Start Speech Thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Load Model with error handling
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize Webcam with error handling
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a file to save the recognized sentence
output_file = 'recognized_sentences.txt'

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary for A-Z
labels_dict = {i: chr(65 + i) for i in range(26)}

# Set expected feature length (must match training)
max_length = 84

# Sentence and Timing Parameters
predicted_sentence = []
letter_interval = 2.0  
pause_interval = 0.20 
last_prediction_time = time.time()
last_predicted_char = None
prev_features = None
static_start_time = None
movement_threshold = 0.005

print("Press 'c' to clear the sentence, or 'q' to exit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    current_predicted_char = None  
    current_features = None         

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))
        
        if len(data_aux) < max_length:
            data_aux += [0] * (max_length - len(data_aux))
        elif len(data_aux) > max_length:
            data_aux = data_aux[:max_length]
        
        current_features = np.array(data_aux).reshape(1, -1)
        
        prediction = model.predict(current_features)
        current_predicted_char = labels_dict.get(int(prediction[0]), "Unknown")
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, current_predicted_char, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    current_time = time.time()
    
    # Static Hand Detection:
    if current_features is not None:
        if prev_features is None:
            prev_features = current_features
            static_start_time = current_time
        else:
            diff = np.mean(np.abs(current_features - prev_features))
            if diff < movement_threshold:
                if static_start_time is None:
                    static_start_time = current_time
                if (current_time - static_start_time) > pause_interval:
                    if predicted_sentence and predicted_sentence[-1] != " ":
                        predicted_sentence.append(" ")
                    static_start_time = current_time  
            else:
                static_start_time = current_time
            prev_features = current_features

    # Debounce logic for letter predictions
    if current_predicted_char is not None and current_predicted_char != "Unknown":
        if current_predicted_char == last_predicted_char:
            if (current_time - last_prediction_time) > letter_interval:
                predicted_sentence.append(current_predicted_char)
                last_prediction_time = current_time
        else:
            predicted_sentence.append(current_predicted_char)
            last_prediction_time = current_time

    if current_predicted_char is not None:
        last_predicted_char = current_predicted_char

    # Display the accumulated sentence
    cv2.putText(frame, "".join(predicted_sentence), (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(50) & 0xFF  
    if key == ord('q'): 
        # Save the recognized sentence to a file
        with open(output_file, 'a') as f:
            f.write(" ".join(predicted_sentence) + "\n")

        sentence_queue.put("EXIT")  # Stop speech thread
        break  
    elif key == ord('c'):
        sentence_queue.put(" ".join(predicted_sentence))  # Speak full sentence before clearing

        predicted_sentence = []  

cap.release()
cv2.destroyAllWindows()
