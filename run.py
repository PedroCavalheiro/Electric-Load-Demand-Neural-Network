import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Configurable parameters
READINGS_PER_HOUR = 4
HOURS_AHEAD = 24 # Horizon
Y_READINGS_AROUND_24H = 26
Z_READINGS_AROUND_7D = 8
A_READINGS_AROUND_14D = 2
HIDDEN_DIMENSIONS = [21, 27]
BATCH_SIZE = 32
DEVICE = "cpu" # small NN, cpu is faster

TRAIN_PERCENTAGE = 0.6
VALID_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

# Calculate the input size
input_size = (Y_READINGS_AROUND_24H + 1) + 1 + (Z_READINGS_AROUND_7D * 2) + 1 + (A_READINGS_AROUND_14D * 2)

class ElectricityLoadPredictor(nn.Module):
    def __init__(self, input_size, hidden_dimensions, hidden_layers):
        super(ElectricityLoadPredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_dimensions[0]))
        for i in range(0, hidden_layers-1):
            self.layers.append(nn.Linear(hidden_dimensions[i], hidden_dimensions[i+1]))
        self.output_layer = nn.Linear(hidden_dimensions[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

class ElectricityLoadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.full(input_size, -1, dtype=np.float32)

        for i in range(Y_READINGS_AROUND_24H + 1):
            day24_idx = idx - (HOURS_AHEAD * READINGS_PER_HOUR) - (i * READINGS_PER_HOUR)
            if 0 <= day24_idx < len(self.data):
                x[Y_READINGS_AROUND_24H - i] = self.data[day24_idx]

        day7_start = idx - (HOURS_AHEAD * READINGS_PER_HOUR * 7)
        if day7_start >= 0:
            z_start = max(day7_start - Z_READINGS_AROUND_7D, 0)
            z_end = min(day7_start + Z_READINGS_AROUND_7D + 1, len(self.data))
            x[Y_READINGS_AROUND_24H + 1:Y_READINGS_AROUND_24H + 1 + (z_end - z_start)] = self.data[z_start:z_end]

        day14_start = idx - (HOURS_AHEAD * READINGS_PER_HOUR * 14)
        if day14_start >= 0:
            a_start = max(day14_start - A_READINGS_AROUND_14D, 0)
            a_end = min(day14_start + A_READINGS_AROUND_14D + 1, len(self.data))
            x[Y_READINGS_AROUND_24H + 1 + (Z_READINGS_AROUND_7D * 2):Y_READINGS_AROUND_24H + 1 + (Z_READINGS_AROUND_7D * 2) + (a_end - a_start)] = self.data[a_start:a_end]

        y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def denormalize_data(data, mean, std):
    return data * std + mean

def plot_predictions(loader, start_idx, end_idx, color, label, no_label=False):
    model.eval()
    true_values = []
    predictions = []
    difference = []

    for idx in range(start_idx, end_idx):
        x, y = loader[idx]
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x.unsqueeze(0))
        true_values.append(denormalize_data(y.cpu().numpy(), mean, std))
        predictions.append(denormalize_data(y_pred.squeeze().detach().cpu().numpy(), mean, std))
        difference.append(denormalize_data((y_pred.squeeze() - y).detach().abs().cpu().numpy(), 0, std))

    plt.plot(range(start_idx, end_idx), true_values, color="blue", linewidth=1, label=None if no_label else "Values", alpha=0.5)
    plt.plot(range(start_idx, end_idx), predictions, color="red", linestyle="--", linewidth=1, label=None if no_label else "Predictions", alpha=0.75)
    plt.plot(range(start_idx, end_idx), difference, color=color, linestyle=":", linewidth=1, label=f"Difference ({label})")

    min_error = np.min(difference)
    max_error = np.max(difference)
    avg_error = np.mean(difference)
    
    return min_error, max_error, avg_error

# Read dataset
data = np.loadtxt('eld180dias.txt', dtype=np.float32)

train_size = int(len(data) * TRAIN_PERCENTAGE)
valid_size = int(len(data) * VALID_PERCENTAGE)
test_size = len(data) - train_size - valid_size

mean = np.mean(data)
std = np.std(data)

# Data normalization
data = (data - mean) / std

# Prepare dataset
dataset = ElectricityLoadDataset(data)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the best model
model = ElectricityLoadPredictor(input_size, HIDDEN_DIMENSIONS, 2).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth"))

# Calculate generalization error
criterion = torch.nn.MSELoss()
model.eval()
with torch.no_grad():
    # Calculate training error
    train_errors = []

    for idx in range(0, train_size):
        x, y = dataset[idx]
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x.unsqueeze(0))
        train_errors.append(mse(y.cpu().numpy(), y_pred.squeeze().detach().cpu().numpy()))

    mean_train_error = np.mean(train_errors)

    # Calculate RMS generalization error
    valid_errors = []

    for idx in range(train_size, train_size + valid_size):
        x, y = dataset[idx]
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x.unsqueeze(0))
        valid_errors.append(mse(y.cpu().numpy(), y_pred.squeeze().detach().cpu().numpy()))

    mean_validation_error = np.mean(valid_errors)

    # Calculate max RMS prediction error
    test_errors = []

    for idx in range(train_size + valid_size, len(dataset)):
        x, y = dataset[idx]
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x.unsqueeze(0))
        test_errors.append(mse(y.cpu().numpy(), y_pred.squeeze().detach().cpu().numpy()))

    mean_test_error = np.mean(test_errors)

print(f"Average Training Error (MSE): {mean_train_error*std}")
print(f"Average Validation Error (MSE): {mean_validation_error*std}")
print(f"Average Test Error (MSE): {mean_test_error*std}")

# Plot the predictions
plt.figure(figsize=(12, 6))
min_train_error, max_train_error, avg_train_error = plot_predictions(dataset, 0, train_size, "blue", "Train")
min_valid_error, max_valid_error, avg_valid_error = plot_predictions(dataset, train_size, train_size + valid_size, "green", "Valid", True)
min_test_error, max_test_error, avg_test_error = plot_predictions(dataset, train_size + valid_size, len(dataset), "red", "Test", True)

train_valid_division = train_size
valid_test_division = train_size + valid_size

plt.axvline(x=train_valid_division, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(x=valid_test_division, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.xlabel("Sample Index")
plt.ylabel("Demand")
plt.title("ELD Values")
plt.legend()
plt.show()
