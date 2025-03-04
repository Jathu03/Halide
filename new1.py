import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
MAX_NODES = 30  # Max number of nodes (functions) in a program
MAX_FEATURES = 37  # 9 scheduling + 8 memory + 20 op histogram
SEQUENCE_LENGTH = MAX_NODES  # Treat each node as a timestep in the sequence

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract features from a single node
def extract_node_features(node):
    features = []
    if "scheduling_feature" in node["Details"]:
        sched = node["Details"]["scheduling_feature"]
        features.extend([
            sched.get("bytes_at_realization", 0.0),
            sched.get("inner_parallelism", 0.0),
            sched.get("outer_parallelism", 0.0),
            sched.get("innermost_loop_extent", 0.0),
            sched.get("num_scalars", 0.0),
            sched.get("num_vectors", 0.0),
            sched.get("points_computed_total", 0.0),
            sched.get("vector_loads_per_vector", 0.0),
            sched.get("working_set_at_root", 0.0),
        ])
    else:
        features.extend([0.0] * 9)

    # Memory access patterns (flatten and truncate/pad to 8 values)
    mem_patterns = node["Details"]["Memory access patterns"]
    mem_values = []
    for pattern in mem_patterns:
        values = [float(x) for x in pattern.split()[-4:]]
        mem_values.extend(values)
    mem_values = mem_values[:8] if len(mem_values) >= 8 else mem_values + [0.0] * (8 - len(mem_values))
    features.extend(mem_values)

    # Operation histogram (truncate/pad to 20 values)
    op_hist = node["Details"]["Op histogram"]
    op_values = []
    for op in op_hist:
        value = float(op.split()[-1])
        op_values.append(value)
    op_values = op_values[:20] if len(op_values) >= 20 else op_values + [0.0] * (20 - len(op_values))
    features.extend(op_values)

    return features

# Process program JSON into a feature tensor and extract execution time
def get_halide_representation(program_dict):
    nodes = program_dict.get("programming_details", {}).get("Nodes", [])
    features_list = []
    exec_time = None

    if not nodes:
        print("Warning: No 'Nodes' found in program_dict")
        return None, None

    for node in nodes:
        if isinstance(node, dict):
            # Check for execution time node
            if node.get("name") == "total_execution_time_ms" and "value" in node:
                exec_time = node["value"] / 1000.0  # Convert ms to seconds
            # Process computation nodes
            elif "Name" in node and "Details" in node:
                node_features = extract_node_features(node)
                features_list.append(node_features)

    if not features_list:
        print("Warning: No computation nodes with 'Name' and 'Details' found")
        return None, exec_time

    # Pad or truncate to MAX_NODES
    if len(features_list) < MAX_NODES:
        features_list.extend([[0.0] * MAX_FEATURES] * (MAX_NODES - len(features_list)))
    elif len(features_list) > MAX_NODES:
        features_list = features_list[:MAX_NODES]

    # Ensure each feature vector is of length MAX_FEATURES
    for i in range(len(features_list)):
        if len(features_list[i]) < MAX_FEATURES:
            features_list[i].extend([0.0] * (MAX_FEATURES - len(features_list[i])))
        elif len(features_list[i]) > MAX_FEATURES:
            features_list[i] = features_list[i][:MAX_FEATURES]

    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    return features_tensor, exec_time

# Load dataset from synthetic_data folder
def load_halide_dataset(data_dir, baseline_time=1.0):
    X_data = []
    y_data = []
    files_processed = 0
    
    # Check if directory exists and contains JSON files
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"No JSON files found in '{data_dir}'.")

    # First pass: determine baseline (max execution time)
    print(f"Scanning {len(json_files)} JSON files for baseline time...")
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            try:
                program_dict = json.load(f)
                features_tensor, exec_time = get_halide_representation(program_dict)
                if exec_time is not None:
                    baseline_time = max(baseline_time, exec_time)
                    files_processed += 1
                else:
                    print(f"Warning: No execution time found in '{filename}'")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in '{filename}'")

    if files_processed == 0:
        raise ValueError(f"No valid execution times found in any JSON files in '{data_dir}'. Check if 'total_execution_time_ms' is present.")

    print(f"Baseline time determined: {baseline_time:.4f} seconds, Files with execution time: {files_processed}")

    # Second pass: compute speedup
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            try:
                program_dict = json.load(f)
                features_tensor, exec_time = get_halide_representation(program_dict)
                if features_tensor is not None and exec_time is not None:
                    speedup = baseline_time / exec_time
                    X_data.append(features_tensor.numpy())
                    y_data.append(speedup)
                else:
                    print(f"Skipping '{filename}': Missing features or execution time")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in '{filename}'")

    if not X_data:
        raise ValueError(f"No valid data points collected from '{data_dir}'. Ensure JSON files contain both computation nodes and execution times.")

    X_data = np.array(X_data)  # Shape: (samples, MAX_NODES, MAX_FEATURES)
    y_data = np.array(y_data)  # Shape: (samples,)
    print(f"Collected {len(X_data)} samples with shape {X_data.shape}")

    # Normalize features
    scaler_X = StandardScaler()
    X_data_flat = X_data.reshape(-1, MAX_FEATURES)
    X_data_scaled = scaler_X.fit_transform(X_data_flat).reshape(X_data.shape)

    scaler_y = StandardScaler()
    y_data_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

    return X_data_scaled, y_data_scaled, scaler_X, scaler_y, baseline_time

# LSTM Model
class LSTMSpeedupPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSpeedupPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last timestep
        return out

# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)
            predictions = model(X_test_tensor).squeeze()
            test_loss = criterion(predictions, y_test_tensor)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Test Loss: {test_loss.item():.4f}")

    return model

# Main function
def predict_halide_speedup(data_dir="synthetic_data"):
    # Load and preprocess data
    X_data, y_data, scaler_X, scaler_y, baseline_time = load_halide_dataset(data_dir)
    print(f"Dataset shape: {X_data.shape}, Baseline time: {baseline_time:.4f} seconds")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Initialize model
    input_size = MAX_FEATURES
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTMSpeedupPredictor(input_size, hidden_size, num_layers, output_size).to(device)

    # Train model
    model = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions_scaled = model(X_test_tensor).cpu().numpy().flatten()
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(y_test_true))):
        print(f"True Speedup: {y_test_true[i]:.4f}, Predicted Speedup: {predictions[i]:.4f}")

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(predictions - y_test_true))
    print(f"Mean Absolute Error: {mae:.4f}")

if __name__ == "__main__":
    predict_halide_speedup(data_dir="synthetic_data")
