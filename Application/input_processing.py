# Application/input_processing.py
import time
import tkinter as tk

class InputProcessor:
    def __init__(self):
        self.typed_text = ""
        self.last_command_time = 0
        self.thesrhold = 0.8
        self.valid_prediction_start_time = None
        self.valid_time = 1
        self.last_input_time = 0
        self.last_input_type = 0
        self.last_sign = None
        self.NORMAL_DETECTION_TIME = 0.8
        self.EXTENDED_DETECTION_TIME = 1.0

    def process_input(self, predictions, app):
        current_time = time.time()

        if predictions and predictions[0]["confidence"] >= self.thesrhold:
            sign = predictions[0]["label"]

            if self.valid_prediction_start_time is None:
                self.valid_prediction_start_time = current_time

            if self.handle_command(sign, current_time, app):
                self.valid_prediction_start_time = current_time
                return True

            required_detection_time = self.NORMAL_DETECTION_TIME
            if (not (len(sign) == 1 and sign.isalnum()) and
                    not sign.startswith('_') and
                    sign == self.last_sign):
                required_detection_time = self.EXTENDED_DETECTION_TIME

            if current_time - self.valid_prediction_start_time >= required_detection_time:
                self.valid_prediction_start_time = current_time

                if current_time - self.last_input_time >= self.valid_time:
                    if len(sign) == 1 and sign.isalnum():
                        if self.last_input_type == 1:
                            self.typed_text += ' '
                        self.typed_text += sign
                        self.last_input_type = 0
                    else:
                        self.typed_text += f" {sign}"
                        self.last_input_type = 1

                    self.last_input_time = current_time
                    self.last_sign = sign

                # Update input field and ensure text is visible
                app.chat_input.delete(0, tk.END)
                app.chat_input.insert(0, self.typed_text)
                app.chat_input.xview_moveto(1.0)  # Scroll to end
                app.chat_input.update_idletasks()  # Force update
                return True
        else:
            self.valid_prediction_start_time = None
            self.last_sign = None

        return False

    def handle_command(self, sign, current_time, app):
        if sign == "_send":
            if current_time - self.last_command_time >= 1:
                app.send_message()
                self.last_command_time = current_time
            return True
        elif sign == "_back":
            if current_time - self.last_command_time >= 1:
                app.delete_last_word()
                self.last_command_time = current_time
            return True
        return False


    def reset_input_state(self):
        self.typed_text = ""
        self.last_command_time = 0
        self.valid_prediction_start_time = None
        self.last_input_time = 0
        self.last_input_type = 0
        self.last_sign = None