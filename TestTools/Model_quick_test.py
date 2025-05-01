import os
import tkinter as tk
from tkinter import Label, Frame
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import torch
import numpy as np
from joblib import load
from Training.model import CNNModel

# Load the trained model and label encoder
MODEL_PATH = r"/Training\model_dir\cnn_asl_model.pth"
ENCODER_PATH = r"/Training\model_dir\cnn_label_encoder.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classes_from_directory(directory):
    """Load class names from the specified directory."""
    class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return len(class_names)

model = CNNModel(num_classes=load_classes_from_directory(r"C:\Users\lowai\PycharmProjects\FYPTest\Preprocessing\augmented_asl_dataset"))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

label_encoder = load(ENCODER_PATH)

mp_hands = mp.solutions.hands


def get_landmarks_from_frame(frame):
    """Extract hand landmarks from the given video frame using MediaPipe."""
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Default landmarks vector
        landmarks = [0] * (21 * 3 * 2)
        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_index >= 2:  # Process up to 2 hands
                    break
                for i, landmark in enumerate(hand_landmarks.landmark):
                    start_idx = hand_index * 21 * 3 + i * 3
                    landmarks[start_idx:start_idx + 3] = [landmark.x, landmark.y, landmark.z]

            # Normalize landmarks
            landmarks = np.array(landmarks)
            if np.max(landmarks) > 0:
                landmarks = landmarks / np.max(landmarks)

            return landmarks.tolist()
    return None


def predict_landmarks(landmarks):
    """Predict the top 3 confidence levels for the given landmarks."""
    if not landmarks:
        return [{"label": "No hands detected", "confidence": 0.0}]

    # Convert landmarks to tensor
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Top-3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
        top_probs = top_probs.squeeze(0).cpu().numpy()
        top_indices = top_indices.squeeze(0).cpu().numpy()

        return [
            {"label": label_encoder.inverse_transform([index])[0], "confidence": float(prob)}
            for index, prob in zip(top_indices, top_probs)
        ]


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("800x600")

        # Video frame
        self.video_frame = Frame(self.root, width=640, height=480, bg="black")
        self.video_frame.pack()

        # Canvas for video feed
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack()

        # Label for predictions
        self.prediction_label = Label(
            self.root,
            text="Top-3 Predictions: ",
            font=("Arial", 14),
            bg="white",
            fg="black",
        )
        self.prediction_label.pack(fill="x")

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Start the video loop
        self.update_video_feed()

    def update_video_feed(self):
        """Update the video feed and process the frame for predictions."""
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            # Resize the frame to match the canvas size (640x480)
            frame = cv2.resize(frame, (640, 480))

            # Extract landmarks and predict
            landmarks = get_landmarks_from_frame(frame)
            predictions = predict_landmarks(landmarks)

            # Update prediction label
            prediction_text = "\n".join(
                f"{pred['label']}: {pred['confidence'] * 100:.2f}%" for pred in predictions
            )
            self.prediction_label.config(text=f"Top-3 Predictions:\n{prediction_text}")

            # Convert the frame to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the frame on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        # Schedule the next frame update
        self.root.after(5, self.update_video_feed)

    def close(self):
        """Release the video capture and close the app."""
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)  # Handle app close
    root.mainloop()