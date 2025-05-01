import speech_recognition as sr
import os
import time

r = sr.Recognizer()

def delete_sound_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error while deleting file: {e}")


def main():
    while True:
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                print("Listening for speech... Start talking now!")

                audio_text = r.listen(source)

                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(audio_text.get_wav_data())

                print("Processing speech...")
                recognized_text = r.recognize_google(audio_text)
                print(f"Text: {recognized_text}")

            delete_sound_file(temp_audio_file)

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google API; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            delete_sound_file("temp_audio.wav")

        time.sleep(2)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()