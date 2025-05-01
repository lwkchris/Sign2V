# Application/App.py
import os
import datetime
import threading
import numpy as np
import cv2
import torch
from joblib import load
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk
from Training.model import CNNModel, FNNModel, MLPModel
from input_processing import InputProcessor
from landmark_display import LandmarkPlotting
from bubble_style import RoundedFrame
from speech_to_text_processing import SpeechProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device name: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

def predict_landmarks(landmarks):
    if not landmarks:
        return [{"label": "No hands detected", "confidence": 0.0}]

    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = app.model(landmarks_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
        top_probs = top_probs.squeeze(0).cpu().numpy()
        top_indices = top_indices.squeeze(0).cpu().numpy()

        return [
            {"label": app.label_encoder.inverse_transform([index])[0], "confidence": float(prob)}
            for index, prob in zip(top_indices, top_probs)
        ]

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.initialize_components()
        self.start_video_capture()
        self.model_var = tk.StringVar(value="CNN")  # Default to CNN
        self.on_model_select(None)  # Load the default model

    def setup_window(self):
        self.root.title("Sign2V : Sign/Speech 2-Ways Conversation")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", lambda e: self.on_closing())

        # Load and set background image
        bg_image = Image.open(r"../Resources/background.png")
        bg_image = bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Create background label
        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # ==== Preset ====
    def preset_chat_messages(self):
        """Pre-set chat messages for the speaker and signer."""
        messages = [
            {"sender": "Speaker", "message": "Hey! Did you got a chance to review the report?", "is_speaker": True},
            {"sender": "Signer", "message": "Yes, I did. I think the report is well-structured.", "is_speaker": False},
            {"sender": "Signer", "message": "However, I feel the conclusion could be stronger.", "is_speaker": False},
            {"sender": "Speaker", "message": "Good point! What do you suggest we add?", "is_speaker": True},
            {"sender": "Signer", "message": "We could emphasize the key findings and their impact.",
             "is_speaker": False},
            {"sender": "Signer", "message": "I can draft a revised version if you'd like.", "is_speaker": False},
            {"sender": "Speaker", "message": "That would be great. Let me know if you need any help.",
             "is_speaker": True},
        ]

        for msg in messages:
            self.add_chat_message(msg["message"], is_speaker=msg["is_speaker"])
    # ==== Preset ====

    def initialize_components(self):
        self.input_processor = InputProcessor()
        self.landmark_plotter = LandmarkPlotting()
        self.create_ui_components()
        self.speech_processor = SpeechProcessor(self)

        # self.preset_chat_messages()

    def create_ui_components(self):
        self.create_banner()
        self.create_right_banner()
        self.create_camera_view()
        self.create_chat_room()

    def create_banner(self):
        banner = tk.Frame(self.root, bg="#333")
        banner.place(relx=0, rely=0, relwidth=1, relheight=0.17)
        banner.lift()  # Ensure it's above background

        self.add_app_icon(banner)
        self.add_title(banner)
        self.add_logo(banner)

    def add_app_icon(self, banner):
        app_icon = Image.open(r"../Resources/icon2.png").resize((115, 115), Image.Resampling.LANCZOS)
        mask = Image.new('L', (115, 115), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, 115, 115], radius=15, fill=255)
        output = Image.new('RGBA', (115, 115), (0, 0, 0, 0))
        output.paste(app_icon, mask=mask)
        app_icon_tk = ImageTk.PhotoImage(output)

        icon_label = tk.Label(banner, image=app_icon_tk, bg="#333")
        icon_label.image = app_icon_tk  # Keep a reference
        icon_label.pack(side=tk.LEFT, padx=20)

    def add_title(self, banner):
        title_container = tk.Frame(banner, bg="#333")
        title_container.pack(side=tk.LEFT, padx=10)

        title_label1 = tk.Label(title_container, text="Sign2V", bg="#333", fg="white", font=("Industry Ultra", 32))
        title_label1.pack(anchor='w')

        title_label2 = tk.Label(title_container, text="Sign/Speech 2-Ways Conversation", bg="#333", fg="white", font=("Industry Ultra", 20))
        title_label2.pack(anchor='e')

    def add_logo(self, banner):
        logo = Image.open(
            r"../Resources/EdUHK_Signature.png").resize((400, 150), Image.Resampling.LANCZOS)
        logo_tk = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(banner, image=logo_tk, bg="#333")
        logo_label.image = logo_tk  # Keep a reference
        logo_label.pack(side=tk.RIGHT, padx=0, pady=0)

    def create_right_banner(self):
        right_banner = tk.Frame(self.root, bg="#FFFFCC")
        right_banner.place(relx=0.87, rely=0.17, relwidth=0.13, relheight=0.83)
        right_banner.lift()  # Ensure it's above background

        self.create_banner_buttons(right_banner)

    def create_banner_buttons(self, right_banner):
        icon_size = (175, 175)
        icons = {
            "send": r"../Resources/Send.png",
            "back": r"../Resources/Back.png",
            "speech": r"../Resources/SpeechStatus.png"
        }

        for idx, (name, path) in enumerate(icons.items()):
            icon = Image.open(path).resize(icon_size, Image.Resampling.LANCZOS)
            icon_tk = ImageTk.PhotoImage(icon)
            button = tk.Button(
                right_banner,
                image=icon_tk,
                bg="#FFFFCC",
                relief=tk.FLAT,
                bd=0,
                activebackground="#FFFFCC"
            )
            button.image = icon_tk
            button.place(relx=0.5, rely=(idx + 0.5) / 3, anchor="center")

    def create_camera_view(self):
        camera_frame = tk.Frame(self.root, bg="white", relief="solid", bd=1)
        camera_frame.place(relx=0.03, rely=0.23, relwidth=0.5, relheight=0.645)
        camera_frame.lift()

        self.camera_label = tk.Label(camera_frame, bg="black")
        self.camera_label.pack(fill="both", expand=True)

        self.create_option_bar(camera_frame)

    def create_option_bar(self, camera_frame):
        option_frame = tk.Frame(camera_frame, bg="#333")
        option_frame.place(relx=0.925, rely=0, relwidth=0.075, relheight=0.03)

        self.model_var = tk.StringVar()

        # Create and configure the combobox style
        style = ttk.Style()
        style.configure('Custom.TCombobox',
                        background='white',
                        fieldbackground='white',
                        foreground='black')

        model_combobox = ttk.Combobox(
            option_frame,
            textvariable=["CNN", "MLP", "FNN"],
            values=["CNN", "MLP", "FNN"],
            style='Custom.TCombobox',
            state="readonly",
            width=5
        )

        # Set the default value
        model_combobox.set("CNN")
        model_combobox.pack(expand=True)

        # Bind the selection event to update the model
        model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        self.create_confidence_bar()

    def delete_last_word(self):
        self.input_processor.typed_text = ' '.join(self.input_processor.typed_text.split()[:-1])
        self.update_input_field(self.input_processor.typed_text)

    def create_confidence_bar(self):
        confidence_bar_frame = tk.Frame(self.root, bg="#f9f9f9", relief="solid", bd=1)
        confidence_bar_frame.place(relx=0.03, rely=0.89, relwidth=0.5, relheight=0.06)
        confidence_bar_frame.lift()  # Ensure it's above background

        self.confidence_canvas = tk.Canvas(confidence_bar_frame, bg="white", height=10, bd=0, highlightthickness=0)
        self.confidence_canvas.pack(fill="both", expand=True, padx=5, pady=3)

    def create_chat_room(self):
        chat_frame = tk.Frame(self.root, bg="white", relief="solid", bd=2)
        chat_frame.place(relx=0.54, rely=0.23, relwidth=0.3, relheight=0.72)
        chat_frame.lift()  # Ensure it's above background

        self.create_chat_controls(chat_frame)
        self.create_chat_display(chat_frame)

    def create_chat_controls(self, chat_frame):
        clear_icon_path = r"../Resources/clear.png"
        clear_icon = Image.open(clear_icon_path).resize((30, 30), Image.Resampling.LANCZOS)
        clear_icon_tk = ImageTk.PhotoImage(clear_icon)

        clear_button = tk.Button(chat_frame, image=clear_icon_tk, bg="white", relief=tk.FLAT, command=self.clear_chat_history)
        clear_button.image = clear_icon_tk
        clear_button.pack(side=tk.TOP, anchor='w', padx=5, pady=5)

        self.input_frame = tk.Frame(chat_frame, bg="white", height=60)
        self.input_frame.pack(fill="x", side="bottom", padx=5, pady=0)
        self.input_frame.pack_propagate(False)

        self.create_input_area()

    def create_input_area(self):
        self.input_container = tk.Frame(self.input_frame, bg="white")
        self.input_container.pack(side=tk.LEFT, fill="x", expand=True)

        signer_label = tk.Label(self.input_container, text="Signer:", bg="white", fg="#333333", font=("Helvetica", 12, "bold"))
        signer_label.pack(side=tk.LEFT, padx=(2, 2), pady=10)

        self.chat_input = tk.Entry(self.input_container, bg="#f1f1f1", fg="#333333", font=("Helvetica", 12), relief=tk.FLAT)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), pady=10)

        self.chat_input.bind('<Return>', self.send_message)

    def create_chat_display(self, chat_frame):
        container_frame = tk.Frame(chat_frame, bg="white")
        container_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container_frame, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

        self.scrollbar = ttk.Scrollbar(container_frame, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.msg_frame = tk.Frame(self.canvas, bg="#ffffff")

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.msg_frame,
            anchor="nw",
            width=self.canvas.winfo_width()
        )

        self.msg_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def start_video_capture(self):

        self.cap = cv2.VideoCapture(0)  # Fallback to the built-in camera

        self.cap.set(cv2.CAP_PROP_FPS, 45)
        self.running = True
        threading.Thread(target=self.update_video_feed, daemon=True).start()

    def update_video_feed(self):
        if not self.running:
            return

        ret, frame = self.cap.read()

        if ret:
            self.process_frame(frame)

        self.root.after(10, self.update_video_feed)  # Schedule to run every 10ms

    def process_frame(self, frame):
        frame = self.resize_frame(frame)
        landmarks, results = LandmarkPlotting.get_landmarks_from_frame(frame)
        predictions = predict_landmarks(landmarks)
        self.landmark_plotter.draw_landmarks(frame, results)
        self.update_confidence_bar(predictions)
        self.input_processor.process_input(predictions, self)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.config(image=imgtk)
        self.camera_label.image = imgtk

    def resize_frame(self, frame):
        # Ensure the camera label has valid dimensions
        window_width = max(1, self.camera_label.winfo_width())
        window_height = max(1, self.camera_label.winfo_height())

        # Validate the input frame
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            # Return a blank frame if the input frame is invalid
            return np.zeros((window_height, window_width, 3), dtype=np.uint8)

        frame_height, frame_width = frame.shape[:2]

        # Calculate scaling factor to fit the height
        scaling_factor = window_height / frame_height
        new_width = int(frame_width * scaling_factor)
        new_height = window_height

        # Initialize a black display frame (to handle padding)
        display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Resize the frame while maintaining aspect ratio
        try:
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Center the resized frame horizontally
            if new_width <= window_width:
                x_padding = (window_width - new_width) // 2
                display_frame[:, x_padding:x_padding + new_width] = resized_frame
            else:
                # Crop the frame horizontally if it exceeds the width
                crop_start = (new_width - window_width) // 2
                display_frame = resized_frame[:, crop_start:crop_start + window_width]
        except cv2.error as e:
            # Log and return a blank frame in case of resize errors
            print(f"Resize error: {e}")
            return display_frame

        # Flip the frame for a mirrored view
        return cv2.flip(display_frame, 1)

    def on_model_select(self, event=None):
        selected_model = self.model_var.get()
        print(f"Selected: {selected_model}")

        model_paths = {
            "CNN": (
                r"../Training/model_dir/cnn_asl_model.pth",
                r"../Training/model_dir/cnn_label_encoder.joblib",
                CNNModel
            ),
            "MLP": (
                r"../Training/model_dir/mlp_asl_model.pth",
                r"../Training/model_dir/mlp_label_encoder.joblib",
                MLPModel
            ),
            "FNN": (
                r"../Training/model_dir/fnn_asl_model.pth",
                r"../Training/model_dir/fnn_label_encoder.joblib",
                FNNModel
            ),
        }

        MODEL_PATH, ENCODER_PATH, model_class = model_paths[selected_model]
        label_list = self.load_label_list()

        print(f"Number of classes: {len(label_list)}")
        self.model = model_class(input_size=21 * 3 * 2, num_classes = len(label_list))
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False
        )
        self.model.to(device)
        self.model.eval()
        self.label_encoder = load(ENCODER_PATH)

    def load_label_list(self):
        reference = r"../Preprocessing/augmented_asl_dataset"
        return [d for d in os.listdir(reference) if os.path.isdir(os.path.join(reference, d))]

    def update_confidence_bar(self, predictions):
        self.confidence_canvas.delete("all")
        total = sum(pred["confidence"] for pred in predictions)

        if total == 0:
            self.confidence_canvas.create_text(
                self.confidence_canvas.winfo_width() // 2,
                self.confidence_canvas.winfo_height() // 2,
                text="Hands not detected",
                fill="black",
                font=("Arial", 12)
            )
            return

        normalized = [pred["confidence"] / total for pred in predictions]
        colors = ["#9C27B0", "#FF9800", "#E0E0E0"]

        x_start = 0
        for i, pred in enumerate(predictions):
            segment_width = normalized[i] * self.confidence_canvas.winfo_width()
            color = colors[i] if i < len(colors) else "gray"
            self.confidence_canvas.create_rectangle(x_start, 10, x_start + segment_width, 30, fill=color, outline="")
            self.confidence_canvas.create_text(
                x_start + 5,
                20,
                text=f"{pred['label']} ({pred['confidence'] * 100:.1f}%)",
                anchor="w",
                fill="white" if color == "#9C27B0" else "black",
                font=("Arial", 12)
            )
            x_start += segment_width

    def on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def clear_chat_history(self):
        for widget in self.msg_frame.winfo_children():
            widget.destroy()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def exit_fullscreen(self, event=None):
        self.on_closing()

    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():
            if event.num == 4 or event.delta > 0:  # Scroll up
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:  # Scroll down
                self.canvas.yview_scroll(1, "units")

    def add_chat_message(self, message, is_speaker=False):
        max_width, min_width, padding, radius = 300, 115, 15, 15
        username = "Speaker" if is_speaker else "Signer"
        timestamp = datetime.datetime.now().strftime("%H:%M")
        color = "#CC00CC" if is_speaker else "#42a5f5"

        temp = tk.Label(self.msg_frame, text=message, font=("Helvetica", 12), wraplength=max_width - 2 * padding)
        temp.pack()
        msg_width = max(temp.winfo_reqwidth() + 2 * padding + 10, min_width)
        msg_height = temp.winfo_reqheight() + padding + 25
        temp.destroy()

        msg_container = tk.Frame(self.msg_frame, bg="#ffffff")
        msg_container.pack(fill=tk.X, pady=3)

        bubble_frame = RoundedFrame(msg_container, bg_color=color, height=msg_height, width=msg_width)
        bubble_frame.pack(side=tk.RIGHT if not is_speaker else tk.LEFT, padx=5)

        bubble_frame.create_rounded_rect(2, 2, msg_width - 2, msg_height - 2, radius, fill=color, outline="", width=0)
        bubble_frame.create_text(padding, 12, text=username, fill="#ffffff", font=("Helvetica", 10, "bold"), anchor='w')
        bubble_frame.create_text(msg_width - padding, 12, text=timestamp, fill="#ffffff", font=("Helvetica", 8),
                                 anchor='e')
        bubble_frame.create_text(
            padding + 2,
            msg_height / 2 + 5,
            text=message.strip(),
            fill="#ffffff",
            font=("Helvetica", 12),
            anchor='w',
            width=max_width - 2 * padding
        )

        self.msg_frame.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def send_message(self, event=None):
        cleaned_text = self.input_processor.typed_text.strip()
        if cleaned_text:
            self.add_chat_message(cleaned_text, is_speaker=False)
            self.input_processor.typed_text = ''
            self.update_input_field('')

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def add_speaker_message(self, message):
        self.add_chat_message(message, is_speaker=True)

    def update_input_field(self, text):
        self.chat_input.delete(0, tk.END)
        self.chat_input.insert(0, text)
        self.chat_input.xview_moveto(1.0)
        self.chat_input.update_idletasks()

    def on_closing(self):
        self.running = False
        self.speech_processor.stop()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()