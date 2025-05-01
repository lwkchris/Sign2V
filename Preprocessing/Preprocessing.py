# Preprocessing/Preprocessing.py
import os
import numpy as np
from tqdm import tqdm
from handmark import MediaPipeProcessor

def preprocess_dataset(dataset_path, save_path="preprocessed_data.npz"):
    processor = MediaPipeProcessor()
    X = []
    y = []
    skipped_count = 0  # Counter for skipped data
    processed_count = 0  # Counter for processed data

    labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]

    # Iterate through dataset using tqdm for progress tracking
    for label in tqdm(labels, desc="Processing Dataset", unit="label"):
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            landmarks = processor.extract_landmarks(file_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
                processed_count += 1
            else:
                print(f"Skipping file due to no landmarks: {file_path}")
                skipped_count += 1

    # Normalize landmarks and encode labels
    X = np.array(X)
    if len(X) > 0:
        X = X / np.max(X)  # Normalize to range [0, 1]
    y = np.array(y)

    # Save preprocessed data
    np.savez(save_path, X=X, y=y)
    print(f"Preprocessed data saved to {save_path}")
    print(f"Processed data points: {processed_count}")
    print(f"Skipped data points: {skipped_count}")

if __name__ == "__main__":
    dataset_path = r"./augmented_asl_dataset"
    save_path = "preprocessed_data.npz"
    preprocess_dataset(dataset_path, save_path)
