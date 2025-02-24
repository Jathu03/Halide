import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.cuda.amp import GradScaler, autocast

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
device = torch.device("cuda")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Load JSON data from a file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Extract features and execution time from a single program
def extract_features(data):
    edges = data["programming_details"]["Edges"]
    nodes = data["programming_details"]["Nodes"]
    scheduling = data["scheduling_data"]
    
    node_dict = {node["Name"]: node["Details"] for node in nodes}
    sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item}
    exec_time = next(item["value"] for item in scheduling if item.get("name") == "total_execution_time_ms")
    
    features = []
    for node_name in node_dict:
        node = node_dict[node_name]
        sched = sched_dict.get(node_name, {})
        
        op_hist = {}
        for entry in node["Op histogram"]:
            parts = entry.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                op_hist[key] = int(value)
        
        prog_features = [
            op_hist.get("Add", 0),
            op_hist.get("Mul", 0),
            op_hist.get("Div", 0),
            op_hist.get("Min", 0),
            op_hist.get("Max", 0),
            op_hist.get("FuncCall", 0),
            len([e for e in edges if e["To"] == node_name or e["To"].startswith(node_name)]),  # Num inputs
            1 if any(e["To"] == f"{node_name}.update(0)" for e in edges) else 0  # Is reduction
        ]
        
        sched_features = [
            sched.get("inner_parallelism", 1.0),
            sched.get("outer_parallelism", 1.0),
            sched.get("vector_size", 16.0),
            sched.get("unrolled_loop_extent", 1.0),
            sched.get("points_computed_total", 0.0),
            sched.get("unique_bytes_read_per_realization", 0.0),
            sched.get("working_set_at_task", 0.0),
            sched.get("innermost_loop_extent", 1.0)
        ]
        
        feature_vector = prog_features + sched_features
        features.append(feature_vector)
    
    return np.array(features), exec_time

# Load and preprocess all synthetic data (keep on CPU until DataLoader)
def load_dataset(data_dir="synthetic_data"):
    X_data = []
    y_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            data = load_data(file_path)
            features, exec_time = extract_features(data)
            
            max_length = 20
            if features.shape[0] < max_length:
                padding = np.zeros((max_length - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))
            elif features.shape[0] > max_length:
                features = features[:max_length]
            
            X_data.append(features)
            y_data.append(exec_time)
    
    X_data = np.array(X_data)  # (samples, timesteps, features)
    y_data = np.array(y_data).reshape(-1, 1)  # (samples, 1)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_flat = X_data.reshape(-1, X_data.shape[-1])
    X_normalized = scaler_X.fit_transform(X_flat).reshape(X_data.shape)
    
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y_data)
    
    # Return as CPU tensors (pin_memory will handle GPU transfer)
    X_tensor = torch.FloatTensor(X_normalized)
    y_tensor = torch.FloatTensor(y_normalized)
    
    return X_tensor, y_tensor, scaler_X, scaler_y

# Define LSTM model
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

# Training function with mixed precision
def train_model(model, X_train, y_train, epochs=100, batch_size=8):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            # Move to GPU inside the loop
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            torch.cuda.empty_cache()

# Generate schedule variations for prediction
def generate_schedule_variations(features, num_variations=5):
    features_cpu = features.cpu().numpy()
    variations = []
    for _ in range(num_variations):
        varied = features_cpu.copy()
        varied[:, 8] *= np.random.uniform(0.5, 2.0)  # inner_parallelism
        varied[:, 9] *= np.random.uniform(0.5, 2.0)  # outer_parallelism
        varied[:, 10] = np.clip(varied[:, 10] * np.random.uniform(0.5, 2.0), 1, 32)  # vector_size
        varied[:, 11] = np.clip(varied[:, 11] * np.random.uniform(0.5, 2.0), 1, 16)  # unrolled_loop_extent
        variations.append(torch.FloatTensor(varied).to(device))
    return variations

# Main function to train and predict
def train_and_predict(data_dir="synthetic_data"):
    X_data, y_data, scaler_X, scaler_y = load_dataset(data_dir)
    print(f"Loaded {X_data.shape[0]} samples with shape {X_data.shape}")
    
    split_idx = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    input_size = X_data.shape[2]
    model = LSTMSpeedupPredictor(input_size).to(device)
    
    train_model(model, X_train, y_train, epochs=100)
    
    model.eval()
    with torch.no_grad():
        X_test_cuda = X_test.to(device)
        y_test_cuda = y_test.to(device)
        with autocast():
            y_pred = model(X_test_cuda)
        test_loss = nn.MSELoss()(y_pred, y_test_cuda)
        print(f"Test Loss (Normalized): {test_loss.item():.4f}")
        
        y_pred_denorm = scaler_y.inverse_transform(y_pred.cpu().numpy())
        y_test_denorm = scaler_y.inverse_transform(y_test.cpu().numpy())
        rmse = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
        print(f"Test RMSE (ms): {rmse:.2f}")
    
    sample_features = X_test[0:1]  # Shape: (1, timesteps, features)
    sample_time = y_test_denorm[0][0]
    print(f"\nBaseline Time for Sample: {sample_time:.2f} ms")
    
    variations = generate_schedule_variations(sample_features[0])
    with torch.no_grad():
        for i, X_var in enumerate(variations):
            X_var = X_var.unsqueeze(0)
            with autocast():
                pred_time_norm = model(X_var).item()
            pred_time = scaler_y.inverse_transform([[pred_time_norm]])[0][0]
            speedup = sample_time / pred_time if pred_time > 0 else 1.0
            print(f"Variation {i+1}: Predicted Time: {pred_time:.2f} ms, Speedup: {speedup:.2f}x")

# Example usage
if __name__ == "__main__":
    train_and_predict(data_dir="synthetic_data")
