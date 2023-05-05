import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

# Number of random experiments
NUM_EXPERIMENTS = 2000

# Keep track of the best model and hyperparameters
best_valid_loss = float("inf")
best_model = None
best_hyperparams = None

# Configurable parameters
READINGS_PER_HOUR = 4
HOURS_AHEAD = 24 # Horizon
Y_READINGS_AROUND_24H = 26
Z_READINGS_AROUND_7D = 8
A_READINGS_AROUND_14D = 2
HIDDEN_LAYERS = 2
HIDDEN_DIMENSIONS = [21, 27]
BATCH_SIZE = 32
LEARNING_RATE = 0.07950075827930025
EPOCHS = 100
PATIENCE = 10
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device {DEVICE}.")
DEVICE = "cpu" # small NN, cpu is faster
TASK_ID = int(sys.argv[1])

TRAIN_PERCENTAGE = 0.6
VALID_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

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

best_hyperparams = {
    "Y_READINGS_AROUND_24H": Y_READINGS_AROUND_24H,
    "Z_READINGS_AROUND_7D": Z_READINGS_AROUND_7D,
    "A_READINGS_AROUND_14D": A_READINGS_AROUND_14D,
    "HIDDEN_DIMENSIONS": HIDDEN_DIMENSIONS,
    "HIDDEN_LAYERS": HIDDEN_LAYERS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
}

with open("best.csv", "w") as file:
    file.write(f"Validation Loss,{','.join(map(str, list(best_hyperparams.keys())))}\n")

for experiment in range(NUM_EXPERIMENTS):
    # Calculate the input size
    input_size = (Y_READINGS_AROUND_24H + 1) + 1 + (Z_READINGS_AROUND_7D * 2) + 1 + (A_READINGS_AROUND_14D * 2)

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
    train_data = ElectricityLoadDataset(data[:train_size])
    valid_data = ElectricityLoadDataset(data[train_size : train_size + valid_size])
    test_data = ElectricityLoadDataset(data[train_size + valid_size:])
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = ElectricityLoadPredictor(input_size, HIDDEN_DIMENSIONS, HIDDEN_LAYERS).to(DEVICE)
    # Train and evaluate the model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    counter = 0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred.squeeze(), y.squeeze())
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        
        # Save the best model and hyperparameters
        if valid_loss < best_valid_loss:
            counter = 0
            best_valid_loss = valid_loss
            best_model = deepcopy(model)
            best_hyperparams = {
                "Y_READINGS_AROUND_24H": Y_READINGS_AROUND_24H,
                "Z_READINGS_AROUND_7D": Z_READINGS_AROUND_7D,
                "A_READINGS_AROUND_14D": A_READINGS_AROUND_14D,
                "HIDDEN_DIMENSIONS": HIDDEN_DIMENSIONS,
                "HIDDEN_LAYERS": HIDDEN_LAYERS,
                "BATCH_SIZE": BATCH_SIZE,
                "LEARNING_RATE": LEARNING_RATE,
            }
            print(f"[{TASK_ID}] New best.\nEpoch {epoch + 1}/{EPOCHS}, Experiment {experiment+1}/{NUM_EXPERIMENTS}, Train Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss}")
            print(valid_loss, best_hyperparams)
            with open("best.csv", "a") as file:
                file.write(f"{valid_loss},{','.join(map(str, list(best_hyperparams.values())))}\n")
            torch.save(best_model.state_dict(), f"best_model_{TASK_ID}.pth")
        else:
            counter += 1
            if counter > PATIENCE:
                # print(f"Skipping experiment after {PATIENCE} epochs without improvement")
                break