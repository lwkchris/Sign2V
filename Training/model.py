# Training/model.py
import torch
import torch.nn as nn

# Swish Activation Function
class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()

        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, 512)  # Input size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.swish = Swish()  # Use Swish activation
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.swish(self.fc1(x))
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.dropout(x)
        x = self.swish(self.fc3(x))
        x = self.dropout(x)
        x = self.swish(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)  # Final layer
        return x


# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.swish = Swish()  # Use Swish activation
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.swish(self.fc1(x))
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# FNN Model
class FNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        self.swish = Swish()  # Use Swish activation
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.swish(self.fc1(x))
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.dropout(x)
        x = self.swish(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x