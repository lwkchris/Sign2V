# Preprocessing/handmark.py
import cv2
import mediapipe as mp

class MediaPipeProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

    def extract_landmarks(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Fixed-size feature vector (21 landmarks * 3 coordinates * 2 hands)
        landmarks = [0] * (21 * 3 * 2)

        with self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            result = hands.process(image)
            if result.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    if hand_index >= 2:  # Ensure no more than 2 hands are processed
                        break
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        start_idx = hand_index * 21 * 3 + i * 3
                        landmarks[start_idx:start_idx + 3] = [landmark.x, landmark.y, landmark.z]
                return landmarks
        return None