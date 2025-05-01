import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from Validating.model import DynamicModel


# Custom Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# Training Function
def train_model(hidden_layers, X, y, epochs, model_name, test_size=0.15, batch_size=256, learning_rate=0.001):
    print(f"\nTraining {model_name} with {len(hidden_layers)} layers and {epochs} epochs...")

    # Encode string labels into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # DataLoader for training
    train_dataset = SignLanguageDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Determine input size and number of classes
    input_size = X.shape[1]
    num_classes = len(np.unique(y_encoded))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with dynamic layers
    model = DynamicModel(hidden_layers, input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    train_loss_values = []  # Store training loss for each epoch
    train_accuracy_values = []  # Store training accuracy for each epoch

    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            for pred, actual in zip(predicted, y_batch):
                if pred == actual:
                    correct += 1

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        train_loss_values.append(avg_train_loss)
        train_accuracy_values.append(train_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

    return train_loss_values, train_accuracy_values


# Plot Training Loss and Accuracy Grouped by Model
def plot_grouped_metrics(results, epochs):
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training loss for all models
    for model_name, data in results.items():
        axes[0].plot(range(1, epochs + 1), data["train_loss"], label=f"{model_name} Training Loss")
    axes[0].set_title("Training Loss Comparison Across Models")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    # Plot training accuracy for all models
    for model_name, data in results.items():
        axes[1].plot(range(1, epochs + 1), data["train_accuracy"], label=f"{model_name} Training Accuracy")
    axes[1].set_title("Training Accuracy Comparison Across Models")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()


# Main Function to Test Multiple Architectures
if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data_path = r"..\Preprocessing\preprocessed_data.npz"

    if os.path.exists(preprocessed_data_path):
        data = np.load(preprocessed_data_path)
        X, y = data["X"], data["y"]
        print(f"Preprocessed data loaded from {preprocessed_data_path}")
    else:
        print(f"No preprocessed data found. Please run preprocessing.py first.")
        exit(1)

    # Define architectures to test
    architectures = {
        "FNN": [512, 256, 128],
        "MLP": [512, 256],
        "CNN": [1024, 512, 256]
    }

    epochs = 400  # Number of epochs to train
    results = {}

    # Train each architecture
    for model_name, hidden_layers in architectures.items():
        print(f"\nTesting {model_name} architecture...")
        train_loss, train_accuracy = train_model(hidden_layers, X, y, epochs, model_name)
        results[model_name] = {"train_loss": train_loss, "train_accuracy": train_accuracy}

    # Plot grouped training loss and accuracy charts
    plot_grouped_metrics(results, epochs)

    # Print the highest accuracy and corresponding epoch for each model
    print("\nHighest Accuracy Points for Each Model:")
    for model_name, data in results.items():
        train_accuracy = data["train_accuracy"]
        highest_accuracy = max(train_accuracy)  # Find the highest accuracy
        best_epoch = train_accuracy.index(highest_accuracy) + 1  # Find the corresponding epoch (1-indexed)
        print(f"Model: {model_name}, Best Epoch: {best_epoch}, "
              f"Highest Accuracy: {highest_accuracy * 100:.2f}%")