import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants (aligned with Tiramisu model where applicable)
MAX_NODES = 20
MAX_LOOPS = 5
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16  # From modeling.py context
MAX_EXPR_LEN = 66  # From data_utils.py

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert Halide graph to tree representation
def halide_to_tree_representation(program_dict):
    nodes = program_dict["programming_details"]["Nodes"]
    edges = program_dict["programming_details"]["Edges"]
    scheduling = program_dict["scheduling_data"]

    node_dict = {node["Name"]: node["Details"] for node in nodes}
    sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item}
    exec_time = next(item["value"] for item in scheduling if item.get("name") == "total_execution_time_ms")

    # Build dependency graph
    dependents = defaultdict(list)
    dependents_of = defaultdict(list)
    for edge in edges:
        from_node, to_node = edge["From"], edge["To"].split(".")[0]
        dependents[from_node].append(to_node)
        dependents_of[to_node].append(from_node)

    # Topological sort for ordering
    def topological_sort():
        in_degree = {n: len(dependents_of[n]) for n in node_dict}
        queue = [n for n in node_dict if in_degree[n] == 0]
        order = []
        while queue and len(order) < MAX_NODES:
            node = queue.pop(0)
            order.append(node)
            for dep in dependents[node]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        return order

    comp_order = topological_sort()

    # Build tree
    tree = {"roots": []}
    seen = set()
    comp_indices_dict = {}
    loop_idx = 0

    for idx, comp_name in enumerate(comp_order):
        if comp_name not in seen:
            comp_indices_dict[comp_name] = idx
            root = {
                "loop_name": f"L{loop_idx}",
                "loop_index": torch.tensor(loop_idx),
                "computations_list": [comp_name],
                "child_list": [],
                "has_comps": True
            }
            loop_idx += 1
            for dep in dependents[comp_name]:
                if dep not in seen and len(seen) < MAX_NODES and loop_idx < MAX_LOOPS:
                    comp_indices_dict[dep] = len(comp_indices_dict)
                    child = {
                        "loop_name": f"L{loop_idx}",
                        "loop_index": torch.tensor(loop_idx),
                        "computations_list": [dep],
                        "child_list": [],
                        "has_comps": True
                    }
                    root["child_list"].append(child)
                    seen.add(dep)
                    loop_idx += 1
            tree["roots"].append(root)
            seen.add(comp_name)
            if len(seen) >= MAX_NODES or loop_idx >= MAX_LOOPS:
                break

    # Assign computation indices
    for root in tree["roots"]:
        root["computations_indices"] = torch.tensor([comp_indices_dict[name] for name in root["computations_list"]])
        for child in root["child_list"]:
            child["computations_indices"] = torch.tensor([comp_indices_dict[name] for name in child["computations_list"]])

    # Computation representation
    comps_repr = []
    for idx, node_name in enumerate(comp_order):
        node = node_dict[node_name]
        sched = sched_dict.get(node_name, {})

        op_hist = {}
        for entry in node["Op histogram"]:
            parts = entry.split(':')
            if len(parts) == 2:
                key, value = parts[0].strip(), int(parts[1].strip().split()[0])
                op_hist[key] = value

        comp_features = [
            op_hist.get("Add", 0),
            op_hist.get("Mul", 0),
            op_hist.get("Div", 0),
            op_hist.get("Min", 0),
            op_hist.get("Max", 0),
            op_hist.get("FuncCall", 0),
            len(dependents_of[node_name]),
            1 if any(e["To"] == f"{node_name}.update(0)" for e in edges) else 0
        ]

        # Transformation tags (placeholder, no detailed transformations in Halide data)
        comp_features.extend([0] * MAX_NUM_TRANSFORMATIONS * MAX_TAGS)

        comps_repr.append(comp_features)

    if len(comps_repr) < MAX_NODES:
        comps_repr.extend([[0] * len(comps_repr[0])] * (MAX_NODES - len(comps_repr)))
    comps_tensor = torch.FloatTensor(comps_repr)
    first_part_size = 8
    comps_tensor_first_part = comps_tensor[:, :first_part_size]
    comps_tensor_vectors = comps_tensor[:, first_part_size:first_part_size + MAX_NUM_TRANSFORMATIONS * MAX_TAGS]
    comps_tensor_third_part = comps_tensor[:, first_part_size + MAX_NUM_TRANSFORMATIONS * MAX_TAGS:]

    # Loops representation
    loops_repr = []
    for loop_idx in range(MAX_LOOPS):
        l_code = f"L{loop_idx}"
        loop_features = [
            1 if sched.get("inner_parallelism", 1.0) > 1.0 else 0,
            1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0,
            sched.get("unrolled_loop_extent", 1.0),
            1 if sched.get("vector_size", 16.0) > 16.0 else 0,
            sched.get("vector_size", 16.0),
            1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0,
            sched.get("unrolled_loop_extent", 1.0)
        ]
        loops_repr.append(loop_features)
    if len(loops_repr) < MAX_LOOPS:
        loops_repr.extend([[0] * 7] * (MAX_LOOPS - len(loops_repr)))
    loops_tensor = torch.FloatTensor(loops_repr)

    # Expressions (placeholder, no expression data in Halide synthetic data)
    functions_comps_expr_tree = torch.zeros(MAX_NODES, MAX_EXPR_LEN, 11)

    return (tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree), exec_time

# Load and preprocess Halide dataset
def load_halide_dataset(data_dir="synthetic_data"):
    X_data = []
    y_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            program_dict = load_data(file_path)
            tree_tensors, exec_time = halide_to_tree_representation(program_dict)
            X_data.append(tree_tensors)
            y_data.append(exec_time)

    y_data = np.array(y_data).reshape(-1, 1)
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y_data)

    return X_data, y_normalized, scaler_y

# Custom Dataset
class HalideDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

# Collate function
def collate_fn(batch):
    trees, comps_first, comps_vectors, comps_third, loops, exprs, y = [], [], [], [], [], [], []
    for tree_tensors, y_val in batch:
        tree, first, vectors, third, loop, expr = tree_tensors
        trees.append(tree)
        comps_first.append(first.numpy())
        comps_vectors.append(vectors.numpy())
        comps_third.append(third.numpy())
        loops.append(loop.numpy())
        exprs.append(expr.numpy())
        y.append(y_val.item())
    return (
        trees,
        torch.from_numpy(np.stack(comps_first, axis=0)),
        torch.from_numpy(np.stack(comps_vectors, axis=0)),
        torch.from_numpy(np.stack(comps_third, axis=0)),
        torch.from_numpy(np.stack(loops, axis=0)),
        torch.from_numpy(np.stack(exprs, axis=0))
    ), torch.from_numpy(np.array(y, dtype=np.float32))

# Model from modeling.py
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=MAX_LOOPS * 7,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size * 2 + loops_tensor_size] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        self.no_comps_tensor = nn.Parameter(torch.zeros(1, embedding_size))
        nn.init.xavier_uniform_(self.no_comps_tensor)
        self.no_nodes_tensor = nn.Parameter(torch.zeros(1, embedding_size))
        nn.init.xavier_uniform_(self.no_nodes_tensor)
        
        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.roots_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.transformation_vectors_embed = nn.LSTM(MAX_TAGS, lstm_embedding_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
        
        nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1) if not nodes_list else self.nodes_lstm(torch.cat(nodes_list, 1))[1][0].permute(1, 0, 2)
        
        if node["has_comps"]:
            indices = node["computations_indices"].to(self.device)
            if indices.max() >= comps_embeddings.shape[1]:
                indices = indices[indices < comps_embeddings.shape[1]]
            if len(indices) > 0:
                selected_comps_tensor = torch.index_select(comps_embeddings, 1, indices)
                comps_h_n = self.comps_lstm(selected_comps_tensor)[1][0].permute(1, 0, 2)
            else:
                comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        
        selected_loop_tensor = torch.index_select(loops_tensor, 1, node["loop_index"].to(self.device))
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors):
        trees, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        batch_size, num_comps, feature_size = comps_tensor_first_part.shape
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device)
        
        vectors = vectors.view(batch_size * num_comps, MAX_NUM_TRANSFORMATIONS, MAX_TAGS)
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1).view(batch_size, num_comps, -1)
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        roots_list = []
        for tree in trees:
            roots_list.extend([self.get_hidden_state(root, comps_embeddings, loops_tensor) for root in tree["roots"]])
        roots_tensor = torch.cat(roots_list, 1)
        lstm_out, (roots_h_n, _) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return self.LeakyReLU(out[:, 0])

# Training function
def train_model(model, train_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for tree_tensors, y_batch in train_loader:
            tree_tensors = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tree_tensors)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(tree_tensors)
            loss = criterion(outputs, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Predict speedup for test data
def predict_halide_speedup(data_dir="synthetic_data"):
    X_data, y_data, scaler_y = load_halide_dataset(data_dir)
    print(f"Loaded {len(X_data)} samples")

    split_idx = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    train_dataset = HalideDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    input_size = X_data[0][1].shape[-1]  # 8 from comp_features
    model = Model_Recursive_LSTM_v2(input_size=input_size, device=device).to(device)
    train_model(model, train_loader, epochs=100)

    model.eval()
    with torch.no_grad():
        y_pred = []
        for tree_tensors in X_test:
            tree_tensors = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tree_tensors)
            pred = model([tree_tensors[0]] + list(tree_tensors[1:]))
            y_pred.append(pred.item())
        y_pred = torch.tensor(y_pred).unsqueeze(-1)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        test_loss = nn.MSELoss()(y_pred.to(device), y_test_tensor)
        print(f"Test Loss (Normalized): {test_loss.item():.4f}")

        y_pred_denorm = scaler_y.inverse_transform(y_pred.numpy())
        y_test_denorm = scaler_y.inverse_transform(y_test)
        rmse = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
        print(f"Test RMSE (ms): {rmse:.2f}")

        print("\nSpeedup Predictions for Test Data:")
        for i in range(min(5, len(y_test_denorm))):
            actual_time = y_test_denorm[i][0]
            pred_time = y_pred_denorm[i][0]
            speedup = actual_time / pred_time if pred_time > 0 else 1.0
            print(f"Test Sample {i+1}: Actual Time: {actual_time:.2f} ms, "
                  f"Predicted Time: {pred_time:.2f} ms, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors
    predict_halide_speedup(data_dir="synthetic_data")
