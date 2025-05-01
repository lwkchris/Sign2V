# Application/landmark_display.py
import cv2
import mediapipe as mp
import numpy as np

class LandmarkPlotting:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def draw_landmarks(self, frame, results):
        if results is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

    def get_landmarks_from_frame(frame):
        with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks = [0] * (21 * 3 * 2)
            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_index >= 2:
                        break
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        start_idx = hand_index * 21 * 3 + i * 3
                        landmarks[start_idx:start_idx + 3] = [landmark.x, landmark.y, landmark.z]

                landmarks = np.array(landmarks)
                if np.max(landmarks) > 0:
                    landmarks = landmarks / np.max(landmarks)

                return landmarks.tolist(), results
        return None, None


