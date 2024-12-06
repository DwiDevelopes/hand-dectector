import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw bounding boxes for detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks: 
            # Get the coordinates of the hand
            h, w, _ = frame.shape
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Draw a circle for each landmark
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw a rectangle around the hand (bounding box)
            x_min = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
            y_min = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
            x_max = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y_max = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()