import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple example model (replace this with your actual model)
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First convolutional layer
        x = F.relu(self.conv2(x))  # Second convolutional layer
        x = F.avg_pool2d(x, kernel_size=4)  # Average pooling
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Function to find the maximum batch size
def find_max_batch_size(model, input_shape, start_batch_size=8, device="cuda"):
    model = model.to(device)  # Move the model to the specified device
    batch_size = start_batch_size
    while True:
        try:
            inputs = torch.randn(batch_size, *input_shape).to(device)
            outputs = model(inputs)
            print(f"Batch size {batch_size}\t fits!")
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Out of memory at batch size {batch_size}")
                return batch_size // 2
            else:
                raise e

# Example usage
if __name__ == "__main__":
    # Check for GPU and print its name
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)  # Get the name of the first GPU
        print(f"Current device: {device} ")
        print(f"- GPU: {gpu_name}")
    else:
        device = "cpu"
        print(f"Current device: {device}")

    # Define an example model
    model = ExampleModel()

    # Specify the input shape (e.g., (3, 224, 224) for an RGB image)
    input_shape = (3, 224, 224)

    # Find the maximum batch size that fits into memory
    max_batch_size = find_max_batch_size(model, input_shape, device=device)
    print(f"The maximum batch size that fits in memory is: {max_batch_size}")