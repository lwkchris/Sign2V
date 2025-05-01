# DataCollecting/DataCollection.py
import cv2
import os
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import time


class HandScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Scanner App")
        self.root.geometry("800x600")

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        # Camera attributes
        self.cap = cv2.VideoCapture(0)  # Automatically start the camera
        self.raw_frame = None  # To store the raw frame without overlays
        self.folder_name = None

        # Tkinter Widgets
        self.folder_label = tk.Label(root, text="Enter Label Name:")
        self.folder_label.pack(pady=5)

        self.folder_entry = tk.Entry(root, width=40)
        self.folder_entry.pack(pady=5)

        # Larger "Capture" button
        self.capture_button = tk.Button(root, text="Capture", command=self.capture_photo, width=20, height=3, bg="lightblue", font=("Arial", 14, "bold"))
        self.capture_button.pack(pady=20)

        self.status_label = tk.Label(root, text="", fg="green", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Start the camera feed immediately
        self.update_frame()

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                self.raw_frame = frame.copy()  # Store the raw frame without overlays
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with MediaPipe
                result = self.hands.process(rgb_frame)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        )

                # Add a grid overlay to the frame
                frame = self.add_grid_overlay(frame)

                # Convert frame to ImageTk format
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

            self.root.after(10, self.update_frame)

    def add_grid_overlay(self, frame):
        """Add a grid overlay to the frame to fix hand position."""
        height, width, _ = frame.shape
        # Draw horizontal lines
        for i in range(1, 3):
            y = height // 3 * i
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        # Draw vertical lines
        for i in range(1, 3):
            x = width // 3 * i
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        return frame

    def capture_photo(self):
        # Get folder name from input
        self.folder_name = self.folder_entry.get().strip()
        if not self.folder_name:
            self.update_status("Error: Enter a label name", "red")
            return

        # Create folder if it doesn't exist
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        if self.raw_frame is not None:
            # Generate a unique filename with timestamp
            timestamp = int(time.time())
            file_name = os.path.join(self.folder_name, f"captured_hand_{timestamp}.jpg")
            # Save the raw frame (without overlays)
            cv2.imwrite(file_name, self.raw_frame)
            self.update_status(f"Photo saved: {file_name}", "green")

    def update_status(self, message, color):
        """Update the status label and clear it after 2 seconds."""
        self.status_label.config(text=message, fg=color)
        self.root.after(2000, self.clear_status)

    def clear_status(self):
        """Clear the status label."""
        self.status_label.config(text="")

    def on_close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.video_label.config(image="")
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandScannerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Handle window close event
    root.mainloop()