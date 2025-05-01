# Application/speech_to_text_processing.py
import speech_recognition as sr
import threading
import queue
import time
from PIL import Image, ImageTk
import tkinter as tk


class SpeechProcessor:
    def __init__(self, app):
        self.app = app
        self.recognizer = sr.Recognizer()
        self.is_listening = True
        self.speech_queue = queue.Queue()

        # Load icons
        self.waiting_icon = Image.open(
            r"../Resources/microphone_waiting.png")
        self.listening_icon = Image.open(
            r"../Resources/microphone_listening.png")

        # Resize icons
        icon_size = (40, 40)
        self.waiting_icon = self.waiting_icon.resize(icon_size, Image.Resampling.LANCZOS)
        self.listening_icon = self.listening_icon.resize(icon_size, Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.waiting_icon_tk = ImageTk.PhotoImage(self.waiting_icon)
        self.listening_icon_tk = ImageTk.PhotoImage(self.listening_icon)

        # Create icon label
        self.mic_label = tk.Label(
            app.input_container,
            image=self.waiting_icon_tk,
            bg='white'
        )
        self.mic_label.pack(side=tk.RIGHT, padx=(5, 10), pady=10)

        self.is_active = False

        # Start all processing threads immediately
        threading.Thread(target=self.process_speech, daemon=True).start()
        threading.Thread(target=self.check_speech_queue, daemon=True).start()
        threading.Thread(target=self.update_status, daemon=True).start()

    def process_speech(self):
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    print("Waiting...")  # Debug print
                    self.is_active = False  # Show waiting icon

                    # Listening for audio
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=20)

                    self.is_active = True  # Show listening icon
                    print("Listening...")  # Debug print
                    text = self.recognizer.recognize_google(audio, language='en-US')
                    print(f"Recognized: {text}")  # Debug print
                    self.speech_queue.put(text)

            except sr.UnknownValueError:
                print("Could not understand audio")
                self.is_active = False
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                self.is_active = False
            except Exception as e:
                print(f"Error: {e}")
                self.is_active = False
            finally:
                self.is_active = False  # Reset to waiting after processing

    def update_status(self):
        while self.is_listening:
            try:
                if self.is_active:
                    self.mic_label.configure(image=self.listening_icon_tk)  # Show listening icon
                else:
                    self.mic_label.configure(image=self.waiting_icon_tk)  # Show waiting icon
            except Exception as e:
                print(f"Error updating status: {e}")  # Debug print
            time.sleep(0.1)

    def check_speech_queue(self):
        while self.is_listening:
            try:
                if not self.speech_queue.empty():
                    text = self.speech_queue.get()
                    self.app.add_speaker_message(text)
            except Exception as e:
                print(f"Error checking speech queue: {e}")  # Debug print
            time.sleep(0.1)

    def stop(self):
        self.is_listening = False
        try:
            self.mic_label.destroy()
        except Exception as e:
            print(f"Error stopping: {e}")  # Debug print