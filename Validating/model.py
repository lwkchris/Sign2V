import torch
import torch.nn as nn

# Swish Activation Function
class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# Dynamic Model Class
class DynamicModel(nn.Module):
    def __init__(self, hidden_layers, input_size, num_classes, activation=Swish, dropout_rate=0.3):
        super(DynamicModel, self).__init__()

        layers = []
        in_features = input_size

        # Dynamically create hidden layers
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))  # Fully connected layer
            layers.append(activation())  # Activation function
            layers.append(nn.Dropout(dropout_rate))  # Dropout
            in_features = hidden_units

        # Add the final output layer
        layers.append(nn.Linear(in_features, num_classes))  # Output layer

        # Use nn.Sequential to bundle all layers together
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)