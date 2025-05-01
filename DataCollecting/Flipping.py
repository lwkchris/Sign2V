# DataCollecting/Flipping.py
import os
from PIL import Image


def flip_images_horizontally(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpeg'):
            input_file_path = os.path.join(folder_path, filename)
            try:
                # Open the image
                with Image.open(input_file_path) as img:
                    # Flip the image horizontally
                    flipped_image = img.transpose(method=Image.FLIP_LEFT_RIGHT)

                    # Save the flipped image (you can change to a different extension if needed)
                    flipped_image.save(input_file_path)
                    print(f"Flipped image saved: {input_file_path}")
            except Exception as e:
                print(f"Error processing file '{input_file_path}': {e}")


if __name__ == "__main__":
    folder_path = input("Enter the path of the folder containing .jpeg images: ")
    flip_images_horizontally(folder_path)