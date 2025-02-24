import json
import numpy as np
from random import randint, uniform, choice

# Helper functions
def random_op_histogram():
    """Generate a random operation histogram."""
    ops = ["Constant", "Cast", "Variable", "Param", "Add", "Sub", "Mul", "Div", "Min", "Max", 
           "EQ", "NE", "LT", "LE", "And", "Or", "Not", "Select", "ImageCall", "FuncCall", "SelfCall", "ExternCall", "Let"]
    hist = {op: 0 for op in ops}
    hist["Variable"] = randint(5, 15)
    hist["Add"] = randint(0, 200)
    hist["Mul"] = randint(0, 100)
    hist["Div"] = randint(0, 10)
    hist["Min"] = randint(0, 5)
    hist["Max"] = randint(0, 5)
    hist["FuncCall"] = randint(0, 50)
    hist["SelfCall"] = randint(0, 2)
    hist["Constant"] = randint(0, 50)
    return [f"      {op}:   {count}" for op, count in hist.items()]

def random_footprint(node_name, dims=3):
    """Generate a random footprint for a node."""
    footprint = []
    for i in range(dims):
        min_bound = f"{node_name}._{i}.min"
        max_bound = f"{node_name}._{i}.max"
        if randint(0, 1):
            min_bound = f"({min_bound} + {randint(-5, 0)})"
            max_bound = f"({max_bound} + {randint(0, 5)})"
        footprint.extend([f"    Min {i}: {min_bound}", f"    Max {i}: {max_bound}"])
    return footprint

def random_jacobian(dims_in, dims_out):
    """Generate a random Load Jacobian matrix."""
    matrix = np.zeros((dims_out, max(dims_in, dims_out)), dtype=int)
    for i in range(min(dims_in, dims_out)):
        if randint(0, 1):
            matrix[i, i] = 1  # Identity-like
        else:
            j = randint(0, dims_in-1)
            matrix[i, j] = randint(1, 8) if randint(0, 1) else 1  # Random scaling or permutation
    return [f" {' '.join(map(str, row))}" for row in matrix]

def random_scheduling_features(node_name, compute_load):
    """Generate scheduling features based on compute load."""
    # Step 1: Define base features
    features = {
        "allocation_bytes_read_per_realization": uniform(0, 100000),
        "bytes_at_production": uniform(1000, 50000000),
        "bytes_at_realization": uniform(1000, 50000000),
        "bytes_at_root": uniform(1000000, 400000000),
        "bytes_at_task": uniform(1000, 1000000),
        "inlined_calls": uniform(0, 1000000),
        "inner_parallelism": choice([1, 2, 4, 8, 16, 32, 64]),
        "outer_parallelism": choice([1, 2, 4, 8, 16, 32, 64]),
        "vector_size": choice([1, 4, 8, 16, 32]),
        "unrolled_loop_extent": choice([1, 2, 4, 8, 12, 16]),
        "points_computed_total": compute_load,
        "unique_bytes_read_per_realization": uniform(0, 10000),
        "working_set_at_task": uniform(0, 2000000),
        "innermost_loop_extent": uniform(1, 5000),
        "num_productions": uniform(1, 100000),
        "num_realizations": uniform(1, 100000),
        "num_scalars": uniform(0, 2000000),
        "num_vectors": uniform(0, 10000000),
    }
    
    # Step 2: Define derived features using base features
    features.update({
        "innermost_bytes_at_production": features["bytes_at_production"] / 1000,
        "innermost_bytes_at_realization": features["bytes_at_realization"] / 1000,
        "innermost_bytes_at_root": features["bytes_at_root"] / 10000,
        "innermost_bytes_at_task": features["bytes_at_task"] / 1000,
        "innermost_pure_loop_extent": features["innermost_loop_extent"],
        "native_vector_size": features["vector_size"],
        "points_computed_minimum": compute_load * 0.8,
        "points_computed_per_production": compute_load / features["num_productions"],
        "points_computed_per_realization": compute_load / features["num_realizations"],
        "scalar_loads_per_scalar": uniform(0, 10),
        "scalar_loads_per_vector": uniform(0, 50),
        "unique_lines_read_per_realization": uniform(0, 100),
        "unique_lines_read_per_task": uniform(0, 500),
        "unique_lines_read_per_vector": uniform(0, 10),
        "vector_loads_per_vector": uniform(0, 10),
        "working_set": uniform(0, 2000000),
        "working_set_at_production": uniform(0, 50000000),
        "working_set_at_realization": uniform(0, 50000000),
        "working_set_at_root": uniform(0, 50000000),
    })
    
    return features

# Generate a single synthetic program
def generate_synthetic_program(sample_id):
    num_nodes = randint(10, 20)
    nodes = []
    edges = []
    scheduling_data = []
    
    # Node names
    node_names = ["input_im"] + [f"func_{i}" for i in range(num_nodes-2)] + ["output"]
    
    # Generate nodes
    for name in node_names:
        dims = 3 if name != "input_im" else 2  # Input might be 2D
        nodes.append({
            "Name": name,
            "Details": {
                "Memory access patterns": ["      Pointwise:      1 0 0 1"] * 4,
                "Op histogram": random_op_histogram(),
                "Region computed": [f"    {name}._{i}.min, {name}._{i}.max" for i in range(dims)],
                "Stage 0": [f"    _{i} {name}._{i}.min {name}._{i}.max" for i in range(dims)],
                "Symbolic region required": [f"    {name}._{i}.min, {name}._{i}.max" for i in range(dims)]
            }
        })
    
    # Generate edges (simple linear pipeline with some branches)
    for i in range(len(node_names)-1):
        from_node = node_names[i]
        to_node = node_names[i+1]
        dims_in = 2 if from_node == "input_im" else 3
        dims_out = 3
        edges.append({
            "Details": {
                "Footprint": random_footprint(to_node, dims_out),
                "Load Jacobians": random_jacobian(dims_in, dims_out)
            },
            "From": from_node,
            "Name": f"{from_node} -> {to_node}",
            "To": to_node
        })
    # Add some reduction updates
    for i in range(1, len(node_names)-1):
        if randint(0, 1):
            edges.append({
                "Details": {
                    "Footprint": random_footprint(node_names[i], 3),
                    "Load Jacobians": random_jacobian(3, 3)
                },
                "From": node_names[i-1],
                "Name": f"{node_names[i-1]} -> {node_names[i]}.update(0)",
                "To": f"{node_names[i]}.update(0)"
            })
    
    # Generate scheduling data
    for name in node_names:
        compute_load = uniform(1000000, 2000000000)  # Random compute load
        scheduling_data.append({
            "Name": name,
            "Details": {"scheduling_feature": random_scheduling_features(name, compute_load)}
        })
    
    # Simulate execution time (simple heuristic)
    total_points = sum(s["Details"]["scheduling_feature"]["points_computed_total"] for s in scheduling_data)
    parallelism = np.mean([s["Details"]["scheduling_feature"]["inner_parallelism"] * 
                          s["Details"]["scheduling_feature"]["outer_parallelism"] for s in scheduling_data])
    memory_load = np.mean([s["Details"]["scheduling_feature"]["bytes_at_root"] for s in scheduling_data])
    exec_time = (total_points / (parallelism * 1000)) + (memory_load / 1e9) * 1000  # ms
    scheduling_data.append({"name": "total_execution_time_ms", "value": exec_time})
    
    return {
        "programming_details": {"Edges": edges, "Nodes": nodes},
        "scheduling_data": scheduling_data
    }

# Generate multiple samples and save
def generate_dataset(num_samples=50, output_dir="synthetic_data"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        program = generate_synthetic_program(i)
        with open(f"{output_dir}/synthetic_{i}.json", "w") as f:
            json.dump(program, f, indent=4)
    print(f"Generated {num_samples} synthetic programs in {output_dir}")

# Example usage
if __name__ == "__main__":
    generate_dataset(num_samples=10000)