import os
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Constants (assumed from your context)
MAX_NODES = 10  # Maximum number of computation nodes (adjust as needed)
MAX_VARS = 5    # Maximum number of variables (similar to MAX_DEPTH)
MAX_TAGS = 8    # Maximum tags per scheduling operation
FEATURE_SIZE = 20  # Define a fixed feature size per node (adjust based on your representation)

def get_halide_representation_template(program_dict, train_device="cpu"):
    funcs_repr_templates_list = []
    funcs_placeholders_indices_dict = {}

    program_json = program_dict["program_annotation"]
    funcs_dict = program_json["functions"]
    ordered_funcs_list = sorted(
        list(funcs_dict.keys()),
        key=lambda x: funcs_dict[x]["definition_order"]
    )

    for func_index, func_name in enumerate(ordered_funcs_list):
        func_dict = funcs_dict[func_name]
        func_repr_template = [float(func_dict["is_pure"])]

        vars_repr = []
        for var_i, var_name in enumerate(func_dict["vars"]):
            v_code = f"V{func_index}-L{var_i}"
            vars_repr.extend([
                v_code + "Split", v_code + "SplitFactor",
                v_code + "Tiled", v_code + "TileFactor",
                v_code + "Parallel", v_code + "Vectorized",
                v_code + "Unrolled", v_code + "UnrollFactor"
            ])
        var_repr_size = 8  # 8 features per variable
        vars_repr.extend([0] * var_repr_size * (MAX_VARS - len(func_dict["vars"])))

        func_repr_template.extend(vars_repr)
        funcs_repr_templates_list.append(func_repr_template)

        for j, element in enumerate(func_repr_template):
            if isinstance(element, str):
                funcs_placeholders_indices_dict[element] = (func_index, j)

    return funcs_repr_templates_list, funcs_placeholders_indices_dict

def get_halide_schedule_representation(program_json, templates, schedule_json, placeholders_dict):
    funcs_repr = [row[:] for row in templates]  # Deep copy of templates
    ordered_funcs_list = sorted(
        list(program_json["functions"].keys()),
        key=lambda x: program_json["functions"][x]["definition_order"]
    )

    for func_index, func_name in enumerate(ordered_funcs_list):
        func_dict = program_json["functions"][func_name]
        func_schedule_dict = schedule_json.get(func_name, {})

        for var_i, var_name in enumerate(func_dict["vars"]):
            v_code = f"V{func_index}-L{var_i}"
            sched = func_schedule_dict.get(var_name, {})

            funcs_repr[func_index][placeholders_dict[v_code + "Split"][1]] = 1 if "split" in sched else 0
            funcs_repr[func_index][placeholders_dict[v_code + "SplitFactor"][1]] = sched.get("split", 0)
            funcs_repr[func_index][placeholders_dict[v_code + "Tiled"][1]] = 1 if "tile" in sched else 0
            funcs_repr[func_index][placeholders_dict[v_code + "TileFactor"][1]] = sched.get("tile", 0)
            funcs_repr[func_index][placeholders_dict[v_code + "Parallel"][1]] = 1 if "parallel" in sched else 0
            funcs_repr[func_index][placeholders_dict[v_code + "Vectorized"][1]] = 1 if "vectorize" in sched else 0
            funcs_repr[func_index][placeholders_dict[v_code + "Unrolled"][1]] = 1 if "unroll" in sched else 0
            funcs_repr[func_index][placeholders_dict[v_code + "UnrollFactor"][1]] = sched.get("unroll", 0)

    # Convert to tensor and pad
    padded_comps = funcs_repr
    if not padded_comps:  # If empty, initialize with zeros
        padded_comps = [[0.0] * FEATURE_SIZE for _ in range(len(ordered_funcs_list))]

    # Ensure consistent row length (FEATURE_SIZE) before padding rows
    for i in range(len(padded_comps)):
        if len(padded_comps[i]) < FEATURE_SIZE:
            padded_comps[i].extend([0.0] * (FEATURE_SIZE - len(padded_comps[i])))
        elif len(padded_comps[i]) > FEATURE_SIZE:
            padded_comps[i] = padded_comps[i][:FEATURE_SIZE]

    # Pad number of nodes (rows) to MAX_NODES
    if len(padded_comps) < MAX_NODES:
        padded_comps.extend([[0.0] * FEATURE_SIZE] * (MAX_NODES - len(padded_comps)))

    comps_tensor = torch.tensor(padded_comps, dtype=torch.float32)
    exec_time = np.min(schedule_json.get("execution_times", [1.0]))  # Default to 1.0 if missing
    return comps_tensor, exec_time

def load_halide_dataset(data_dir):
    X_data = []
    y_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r") as f:
                program_dict = json.load(f)
                templates, placeholders_dict = get_halide_representation_template(program_dict)
                for schedule in program_dict.get("schedules_list", []):
                    comps_tensor, exec_time = get_halide_schedule_representation(
                        program_dict["program_annotation"], templates, schedule, placeholders_dict
                    )
                    X_data.append(comps_tensor.flatten().numpy())
                    y_data.append(exec_time)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_data = scaler_X.fit_transform(X_data)
    y_data = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

    return X_data, y_data, scaler_X, scaler_y

def predict_halide_speedup(data_dir):
    X_data, y_data, scaler_X, scaler_y = load_halide_dataset(data_dir)
    # Placeholder for model training/prediction (not implemented here)
    print("Dataset loaded successfully:", X_data.shape, y_data.shape)

if __name__ == "__main__":
    predict_halide_speedup(data_dir="synthetic_data")
