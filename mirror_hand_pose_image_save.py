import cv2
import numpy as np
import mediapipe as mp
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set flag to improve performance
        image.flags.writeable = False

        # Perform hand detection
        results = hands.process(image)

        # Convert RGB back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Print detection results (optional)
        print(results)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 256, 0), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(0, 0, 256), thickness=2, circle_radius=2))


        #Save images // uncomment if you need to save images
        # cv2.imwrite(os.path.join('save_images','{}.jpg'.format(uuid.uuid1())),image)


        # Display the resulting frame
        cv2.imshow('Hand Tracking (Mirrored)', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
