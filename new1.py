import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for representation
MAX_NODES = 20  # Max number of nodes (computations)
MAX_LOOPS = 5   # Max loop depth
MAX_TRANSFORMS = 4  # Max number of transformations per computation
MAX_TAGS = 8    # Size of transformation tag vector

# Load JSON data from a file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Create a template for Halide program representation
def get_halide_representation_template(program_dict):
    nodes = program_dict["programming_details"]["Nodes"]
    edges = program_dict["programming_details"]["Edges"]
    scheduling = program_dict["scheduling_data"]

    node_dict = {node["Name"]: node["Details"] for node in nodes}
    sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item}

    comps_repr_templates = []
    comps_indices_dict = {}
    comps_placeholders_indices_dict = {}

    for comp_idx, node_name in enumerate(node_dict.keys()):
        node = node_dict[node_name]
        sched = sched_dict.get(node_name, {})

        op_hist = {}
        for entry in node["Op histogram"]:
            parts = entry.split(':')
            if len(parts) == 2:
                key, value = parts[0].strip(), int(parts[1].strip().split()[0])
                op_hist[key] = value

        comp_repr = [
            op_hist.get("Add", 0),
            op_hist.get("Mul", 0),
            op_hist.get("Div", 0),
            op_hist.get("Min", 0),
            op_hist.get("Max", 0),
            op_hist.get("FuncCall", 0),
            len([e for e in edges if e["To"] == node_name or e["To"].startswith(node_name)]),  # Inputs
            1 if any(e["To"] == f"{node_name}.update(0)" for e in edges) else 0  # Reduction
        ]

        loop_repr = []
        c_code = f"C{comp_idx}"
        for loop_idx in range(MAX_LOOPS):
            l_code = f"{c_code}-L{loop_idx}"
            loop_repr.extend([
                f"{l_code}-Parallel",
                f"{l_code}-Tile",
                f"{l_code}-TileFactor",
                f"{l_code}-Vectorize",
                f"{l_code}-VectorSize",
                f"{l_code}-Unroll",
                f"{l_code}-UnrollFactor"
            ])
        comp_repr.extend(loop_repr)

        comp_repr.append(f"{c_code}-TransformTagsStart")
        comp_repr.extend(["T"] * (MAX_TRANSFORMS * MAX_TAGS - 2))
        comp_repr.append(f"{c_code}-TransformTagsEnd")

        comps_repr_templates.append(comp_repr)
        comps_indices_dict[node_name] = comp_idx
        for j, element in enumerate(comp_repr):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_idx, j)

    return comps_repr_templates, comps_indices_dict, comps_placeholders_indices_dict

# Fill the template with schedule-specific features
def get_halide_schedule_representation(program_dict, comps_repr_templates, comps_indices_dict, comps_placeholders_indices_dict):
    nodes = program_dict["programming_details"]["Nodes"]
    scheduling = program_dict["scheduling_data"]
    node_dict = {node["Name"]: node["Details"] for node in nodes}
    sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item}
    exec_time = next(item["value"] for item in scheduling if item.get("name") == "total_execution_time_ms")

    comps_repr = [list(template) for template in comps_repr_templates]

    for comp_idx, node_name in enumerate(node_dict.keys()):
        sched = sched_dict.get(node_name, {})
        c_code = f"C{comp_idx}"

        for loop_idx in range(min(MAX_LOOPS, 2)):  # Assume 2D loops (x, y)
            l_code = f"{c_code}-L{loop_idx}"
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-Parallel"][1]] = sched.get("inner_parallelism", 1.0) > 1.0
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-Tile"][1]] = 1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-TileFactor"][1]] = sched.get("unrolled_loop_extent", 1.0)
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-Vectorize"][1]] = 1 if sched.get("vector_size", 16.0) > 16.0 else 0
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-VectorSize"][1]] = sched.get("vector_size", 16.0)
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-Unroll"][1]] = 1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0
            comps_repr[comp_idx][comps_placeholders_indices_dict[f"{l_code}-UnrollFactor"][1]] = sched.get("unrolled_loop_extent", 1.0)

        tags = [0] * (MAX_TRANSFORMS * MAX_TAGS)
        tags_start = comps_placeholders_indices_dict[f"{c_code}-TransformTagsStart"]
        tags_end = comps_placeholders_indices_dict[f"{c_code}-TransformTagsEnd"]
        comps_repr[comp_idx][tags_start[1]:tags_end[1] + 1] = tags

    padded_comps = []
    for comp in comps_repr:
        padded_comps.append([float(x) if not isinstance(x, str) else 0.0 for x in comp])
    if len(padded_comps) < MAX_NODES:
        padded_comps.extend([[0.0] * len(padded_comps[0])] * (MAX_NODES - len(padded_comps)))
    elif len(padded_comps) > MAX_NODES:
        padded_comps = padded_comps[:MAX_NODES]

    return torch.FloatTensor(padded_comps).unsqueeze(0), float(exec_time)

# Load and preprocess Halide dataset
def load_halide_dataset(data_dir="synthetic_data"):
    X_data = []
    y_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            program_dict = load_data(file_path)
            templates, _, placeholders_dict = get_halide_representation_template(program_dict)
            comps_tensor, exec_time = get_halide_schedule_representation(program_dict, templates, {}, placeholders_dict)
            X_data.append(comps_tensor.squeeze(0).numpy())
            y_data.append(exec_time)

    X_data = np.array(X_data)  # Shape: (samples, MAX_NODES, features)
    y_data = np.array(y_data).reshape(-1, 1)  # Shape: (samples, 1)

    # Normalize
    scaler_X = MinMaxScaler()
    X_flat = X_data.reshape(-1, X_data.shape[-1])
    X_normalized = scaler_X.fit_transform(X_flat).reshape(X_data.shape)
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y_data)

    return X_normalized, y_normalized, scaler_X, scaler_y

# LSTM Model
class LSTMSpeedupPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMSpeedupPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training function
def train_model(model, X_train, y_train, epochs=100, batch_size=8):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Predict speedup for test data
def predict_halide_speedup(data_dir="synthetic_data"):
    # Load and preprocess data
    X_data, y_data, scaler_X, scaler_y = load_halide_dataset(data_dir)
    print(f"Loaded {X_data.shape[0]} samples with shape {X_data.shape}")

    # Train-test split (80% train, 20% test)
    split_idx = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Initialize and train model
    input_size = X_data.shape[2]
    model = LSTMSpeedupPredictor(input_size).to(device)
    train_model(model, X_train, y_train, epochs=100)

    # Predict on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        y_pred = model(X_test_tensor)
        test_loss = nn.MSELoss()(y_pred, y_test_tensor)
        print(f"Test Loss (Normalized): {test_loss.item():.4f}")

        # Denormalize predictions and actual values
        y_pred_denorm = scaler_y.inverse_transform(y_pred.cpu().numpy())
        y_test_denorm = scaler_y.inverse_transform(y_test)
        rmse = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
        print(f"Test RMSE (ms): {rmse:.2f}")

        # Compute speedups for test data
        print("\nSpeedup Predictions for Test Data:")
        for i in range(min(5, len(y_test_denorm))):  # Show first 5 test samples
            actual_time = y_test_denorm[i][0]
            pred_time = y_pred_denorm[i][0]
            speedup = actual_time / pred_time if pred_time > 0 else 1.0
            print(f"Test Sample {i+1}: Actual Time: {actual_time:.2f} ms, "
                  f"Predicted Time: {pred_time:.2f} ms, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    predict_halide_speedup(data_dir="c1")
