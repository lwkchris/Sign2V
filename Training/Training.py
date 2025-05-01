# Training/Training.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from joblib import dump
from matplotlib import pyplot as plt
from model import CNNModel, MLPModel, FNNModel

# Custom Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# CNN Training
def CNN_train_model(X, y, random_state, epoch, test_size, model_dir="model_dir"):
    model_save_path = os.path.join(model_dir, "cnn_asl_model.pth")
    label_encoder_save_path = os.path.join(model_dir, "cnn_label_encoder.joblib")

    # Encode string labels into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Unique classes in training data: {len(np.unique(y_encoded))}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # DataLoader for training and testing
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Determine number of classes
    num_classes = len(np.unique(y_encoded))
    input_size = X.shape[1]  # Input size is the number of features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the CNN model
    model = CNNModel(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    model.train()

    loss_values = []  # To store loss for each epoch
    for ep in range(epoch):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_loss)  # Save epoch loss
        print(f"Epoch [{ep + 1}/{epoch}], Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.topk(outputs, 3, dim=1)  # Get top 3 predictions
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # Calculate accuracy
    y_pred_flat = np.array([p[0] for p in y_pred])
    CNN_accuracy = np.mean(y_pred_flat == y_true)
    print(f"Model Accuracy: {CNN_accuracy * 100:.2f}%")

    # Save the model and label encoder
    torch.save(model.state_dict(), model_save_path)
    dump(label_encoder, label_encoder_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"LabelEncoder saved to {label_encoder_save_path}")

    return CNN_accuracy, loss_values  # Return accuracy and loss values

# MLP Training
def MLP_train_model(X, y, random_state, epoch, test_size, model_dir="model_dir"):
    model_save_path = os.path.join(model_dir, "mlp_asl_model.pth")
    label_encoder_save_path = os.path.join(model_dir, "mlp_label_encoder.joblib")

    # Encode string labels into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Unique classes in training data: {len(np.unique(y_encoded))}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # DataLoader for training and testing
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]  # Input size is the number of features
    num_classes = len(np.unique(y_encoded))

    # Initialize the MLP model
    model = MLPModel(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    loss_values = []  # To store loss for each epoch
    for ep in range(epoch):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_loss)  # Save epoch loss
        print(f"Epoch [{ep + 1}/{epoch}], Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.topk(outputs, 3, dim=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # Calculate accuracy
    y_pred_flat = np.array([p[0] for p in y_pred])
    accuracy = np.mean(y_pred_flat == y_true)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model and label encoder
    torch.save(model.state_dict(), model_save_path)
    dump(label_encoder, label_encoder_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"LabelEncoder saved to {label_encoder_save_path}")

    return accuracy, loss_values  # Return accuracy and loss values

# FNN Training (Similar to MLP)
def FNN_train_model(X, y, random_state, epoch, test_size, model_dir="model_dir"):
    model_save_path = os.path.join(model_dir, "fnn_asl_model.pth")
    label_encoder_save_path = os.path.join(model_dir, "fnn_label_encoder.joblib")

    # Encode string labels into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Unique classes in training data: {len(np.unique(y_encoded))}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # DataLoader for training and testing
    train_dataset = SignLanguageDataset(X_train, y_train)
    test_dataset = SignLanguageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[1]  # Input size is the number of features
    num_classes = len(np.unique(y_encoded))

    # Initialize the FNN model
    model = FNNModel(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    loss_values = []  # To store loss for each epoch
    for ep in range(epoch):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_loss)  # Save epoch loss
        print(f"Epoch [{ep + 1}/{epoch}], Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.topk(outputs, 3, dim=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # Calculate accuracy
    y_pred_flat = np.array([p[0] for p in y_pred])
    accuracy = np.mean(y_pred_flat == y_true)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model and label encoder
    torch.save(model.state_dict(), model_save_path)
    dump(label_encoder, label_encoder_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"LabelEncoder saved to {label_encoder_save_path}")

    return accuracy, loss_values  # Return accuracy and loss values

# Main function
if __name__ == "__main__":
    # Define model directory
    os.makedirs("./model_dir", exist_ok=True)

    # Load preprocessed data
    preprocessed_data_path = r"..\Preprocessing\preprocessed_data.npz"
    epoch = 330
    random_state = 42
    test_size = 0.15

    if os.path.exists(preprocessed_data_path):
        data = np.load(preprocessed_data_path)
        X, y = data["X"], data["y"]
        print(f"Preprocessed data loaded from {preprocessed_data_path}")
    else:
        print(f"No preprocessed data found at {preprocessed_data_path}. Please run preprocessing.py first.")
        exit(1)

    # Select which model to train using match...case
    print("Select model to train:")
    print("0. CNN")
    print("1. MLP")
    print("2. FNN")
    print("3. All Models")
    choice = int(input("Enter your choice (0/1/2/3): "))

    accuracies = {}
    loss_curves = {}

    match choice:
        case 0:
            print("\nTraining CNN...")
            cnn_accuracy, cnn_loss_values = CNN_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["CNN"] = cnn_accuracy
            loss_curves["CNN"] = cnn_loss_values

        case 1:
            print("\nTraining MLP...")
            mlp_accuracy, mlp_loss_values = MLP_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["MLP"] = mlp_accuracy
            loss_curves["MLP"] = mlp_loss_values

        case 2:
            print("\nTraining FNN...")
            fnn_accuracy, fnn_loss_values = FNN_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["FNN"] = fnn_accuracy
            loss_curves["FNN"] = fnn_loss_values

        case 3:
            print("\nTraining CNN...")
            cnn_accuracy, cnn_loss_values = CNN_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["CNN"] = cnn_accuracy
            loss_curves["CNN"] = cnn_loss_values

            print("\nTraining MLP...")
            mlp_accuracy, mlp_loss_values = MLP_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["MLP"] = mlp_accuracy
            loss_curves["MLP"] = mlp_loss_values

            print("\nTraining FNN...")
            fnn_accuracy, fnn_loss_values = FNN_train_model(X, y, random_state, epoch, test_size, "model_dir")
            accuracies["FNN"] = fnn_accuracy
            loss_curves["FNN"] = fnn_loss_values

        case _:
            print("Invalid choice. Please select 0, 1, 2, or 3.")
            exit(1)

    # Plotting Loss and Comparison
    print("\nPlotting Loss Curves...")
    plt.figure(figsize=(10, 6))

    # Plot each model's loss
    for model_name, losses in loss_curves.items():
        plt.plot(losses, label=f"{model_name} Loss", linewidth=2)

    # Add title and labels
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Customize the y-axis scale to show finer details (e.g., step of 0.1)
    plt.yticks(ticks=np.arange(0, max(max(losses) for losses in loss_curves.values()) + 0.1, 0.1))

    # Add legend and grid
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

    print("\nModel Performance Comparison:")
    for model_name, accuracy in accuracies.items():
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
