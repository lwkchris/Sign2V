# Preprocessing/DataAugmentation.py
import os
import random
from shutil import copy2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class Augmentor:
    def __init__(self, input_folder, output_folder, augmented_per_folder=200):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.augmented_per_folder = augmented_per_folder
        self.label_count = 0

        self.transform = transforms.Compose([
            transforms.Pad(padding=50, fill=(0, 0, 0)),                         # Padding
            transforms.Resize((512, 512)),                                      # Resizing to 512x512
            transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.05)),    # Rotation
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

        self.augment_dataset()

    @staticmethod
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def validate_paths(self):
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError(f"Input folder '{self.input_folder}' does not exist.")
        self.ensure_dir_exists(self.output_folder)

    def copy_or_augment_folder(self, root, files):
        relative_path = os.path.relpath(root, self.input_folder)
        output_dir = os.path.join(self.output_folder, relative_path)
        self.ensure_dir_exists(output_dir)

        # Filter valid image files in the current folder
        image_files = [file for file in files if file.lower().endswith(('.jpeg', '.jpg', '.png'))]

        if len(image_files) == 0:
            print(f"Skipping folder '{relative_path}' as it contains no valid image files.")
            return  # Skip folders with no valid images

        # Check if the output folder already has enough images
        existing_augmented_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if len(existing_augmented_files) >= self.augmented_per_folder:
            print(f"Folder '{relative_path}' already has {len(existing_augmented_files)} files, skipping.")
            return

        # Copy the existing files
        for file in tqdm(image_files, desc=f"Copying Label [{relative_path}]", unit="file"):
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_dir, file)
            try:
                copy2(input_file_path, output_file_path)
            except Exception as e:
                print(f"Error copying file '{input_file_path}': {e}")

        # Calculate the number of images needed
        remaining = self.augmented_per_folder - len(existing_augmented_files) - len(image_files)

        # Generate the remaining augmented images
        for i in tqdm(range(remaining), desc=f"Augmenting Label [{relative_path}]", unit="image"):
            random_file = random.choice(image_files)
            input_file_path = os.path.join(root, random_file)
            file_name, file_extension = os.path.splitext(random_file)

            try:
                image = Image.open(input_file_path).convert('RGB')
                augmented_image = self.transform(image)

                output_file_name = f"{file_name}_augmented_{i + 1}{file_extension}"
                output_file_path = os.path.join(output_dir, output_file_name)
                augmented_image.save(output_file_path)
            except Exception as e:
                print(f"Error processing file '{input_file_path}': {e}")

        # Increment the label count after processing each folder
        self.label_count += 1

    def augment_dataset(self):
        self.validate_paths()
        labels = []

        for root, dirs, files in os.walk(self.input_folder):
            if root == self.input_folder:
                labels.extend(dirs)
            self.copy_or_augment_folder(root, files)

        total_labels = len(labels)
        print(f"\nLabels:\n {labels}\n")
        print(f"Total labels (subdirectories) found: {total_labels}")
        print(f"Total number of labels processed: {self.label_count}")


if __name__ == "__main__":
    print("Starting data augmentation...")
    augmentor = Augmentor(input_folder='./asl_dataset',
                          output_folder=r"./augmented_asl_dataset",
                          augmented_per_folder=200)
    print("Data augmentation completed and saved.")