import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Webcam
cap = cv2.VideoCapture(0)  # Adjust camera index if necessary
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary for A-Z
labels_dict = {i: chr(65 + i) for i in range(26)}

# Set expected feature length (must match training)
max_length = 84  # Adjust according to your training

# Initialize sentence and debounce parameters
predicted_sentence = ""
letter_interval = 2.0  # seconds: minimum gap before appending the same letter repeatedly
# We'll use pause_interval for detecting a deliberate pause based on static hand movement.
pause_interval = 0.30   # seconds: if the hand remains almost unchanged for this duration, insert a space

last_prediction_time = time.time()
last_predicted_char = None

# For static detection, store previous feature vector and time of last change.
prev_features = None
static_start_time = None
movement_threshold = 0.005  # Adjust: lower means stricter; this is an example value.

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
    
    current_predicted_char = None  # Reset for each frame
    current_features = None         # To store current feature vector if computed
    
    if results.multi_hand_landmarks:
        # Draw landmarks for visual feedback
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Use the first detected hand for prediction
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))
        
        # Pad or truncate data_aux to match max_length
        if len(data_aux) < max_length:
            data_aux += [0] * (max_length - len(data_aux))
        elif len(data_aux) > max_length:
            data_aux = data_aux[:max_length]
        
        current_features = np.array(data_aux).reshape(1, -1)
        
        # Get model prediction
        prediction = model.predict(current_features)
        current_predicted_char = labels_dict.get(int(prediction[0]), "Unknown")
        
        # Optional: draw a bounding box around the hand for feedback
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
            # Compute average absolute difference between current and previous features.
            diff = np.mean(np.abs(current_features - prev_features))
            if diff < movement_threshold:
                # Hand is almost static.
                if static_start_time is None:
                    static_start_time = current_time
                # If static for longer than pause_interval, insert a space.
                if (current_time - static_start_time) > pause_interval:
                    if predicted_sentence == "" or predicted_sentence[-1] != " ":
                        predicted_sentence += " "
                    static_start_time = current_time  # reset to avoid continuous space insertion
            else:
                # Hand has moved significantly, update the static timer.
                static_start_time = current_time
            prev_features = current_features

    # Debounce logic for letter predictions:
    if current_predicted_char is not None and current_predicted_char != "Unknown":
        # If the same letter continues and the letter_interval has passed, add it.
        if current_predicted_char == last_predicted_char:
            if (current_time - last_prediction_time) > letter_interval:
                predicted_sentence += current_predicted_char
                last_prediction_time = current_time
        else:
            # Different letter: append immediately and update time.
            predicted_sentence += current_predicted_char
            last_prediction_time = current_time
    
    # Update last predicted character if available
    if current_predicted_char is not None:
        last_predicted_char = current_predicted_char

    # Display the accumulated sentence at the top of the frame
    cv2.putText(frame, predicted_sentence, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(50) & 0xFF  # Slight delay to slow down capture
    if key == ord('q'):
        break  # Exit on 'q'
    elif key == ord('c'):
        predicted_sentence = ""  # Clear sentence on 'c'

cap.release()
cv2.destroyAllWindows()
