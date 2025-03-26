import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

def extract_features_from_json(json_data):
    """
    Extract features from a single JSON file containing Halide program data.
    
    Args:
        json_data (dict): Loaded JSON data
        
    Returns:
        dict: Extracted features
    """
    features = {}
    
    # Extract execution time
    for item in json_data['programming_details'].get('Scheduling', []):
        if item.get('name') == 'total_execution_time_ms':
            features['execution_time_ms'] = item['value']
            break
    
    # Extract Edge features
    edges = json_data['programming_details']['Edges']
    features['num_edges'] = len(edges)
    
    # Aggregate Load Jacobian statistics
    jacobian_sizes = []
    for edge in edges:
        jacobians = edge['Details']['Load Jacobians']
        jacobian_sizes.append(len(jacobians))
    features['avg_jacobian_size'] = np.mean(jacobian_sizes) if jacobian_sizes else 0
    features['max_jacobian_size'] = max(jacobian_sizes) if jacobian_sizes else 0
    
    # Extract Node features
    nodes = json_data['programming_details']['Nodes']
    features['num_nodes'] = len(nodes)
    
    # Aggregate operation histogram statistics
    op_counts = {'Add': 0, 'Mul': 0, 'Div': 0, 'Min': 0, 'Max': 0}
    for node in nodes:
        op_hist = node['Details']['Op histogram']
        for op_line in op_hist:
            op_name = op_line.split(':')[0].strip().split()[-1]
            count = int(op_line.split(':')[-1].strip())
            if op_name in op_counts:
                op_counts[op_name] += count
    for op_name, count in op_counts.items():
        features[f'total_{op_name.lower()}_ops'] = count
    
    # Extract Scheduling features
    scheduling_nodes = json_data['programming_details'].get('Scheduling', [])
    scheduling_features = {}
    
    for node in scheduling_nodes:
        if 'Details' in node and 'scheduling_feature' in node['Details']:
            sf = node['Details']['scheduling_feature']
            node_name = node['Name']
            for key, value in sf.items():
                scheduling_features[f"{key}_{node_name}"] = value
    
    # Aggregate scheduling features across nodes
    agg_features = {
        'bytes_at_root': [],
        'points_computed_total': [],
        'num_vectors': [],
        'working_set_at_root': [],
        'vector_loads_per_vector': [],
        'inner_parallelism': [],
        'outer_parallelism': []
    }
    
    for key, value in scheduling_features.items():
        for feat in agg_features.keys():
            if feat in key and isinstance(value, (int, float)):
                agg_features[feat].append(value)
    
    # Add aggregated scheduling features
    for feat, values in agg_features.items():
        if values:
            features[f'avg_{feat}'] = np.mean(values)
            features[f'max_{feat}'] = max(values)
            features[f'total_{feat}'] = sum(values)
    
    return features

def process_halide_programs(root_dir="synthetic_data"):
    """
    Process Halide programs from synthetic_data folder and create a dataset.
    
    Args:
        root_dir (str): Root directory containing Halide program subfolders
        
    Returns:
        DataFrame: Dataset with extracted features
    """
    dataset = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory {root_dir} not found")
    
    # Iterate through program folders
    for program_folder in root_path.iterdir():
        if program_folder.is_dir():
            program_name = program_folder.name
            
            # Process each schedule file
            for schedule_file in program_folder.iterdir():
                if schedule_file.is_file() and schedule_file.suffix == '.json':
                    schedule_name = schedule_file.stem
                    
                    try:
                        with open(schedule_file, 'r') as f:
                            json_data = json.load(f)
                        
                        # Extract features
                        features = extract_features_from_json(json_data)
                        features['program_name'] = program_name
                        features['schedule_name'] = schedule_name
                        features['file_path'] = str(schedule_file)
                        
                        dataset.append(features)
                        
                    except Exception as e:
                        print(f"Error processing {schedule_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    return df

def print_dataset_summary(df):
    """
    Print summary of the processed dataset.
    
    Args:
        df (DataFrame): Processed dataset
    """
    print(f"Total samples: {len(df)}")
    print(f"Unique programs: {df['program_name'].nunique()}")
    print(f"Features extracted: {len(df.columns)}")
    print("\nFeature columns:")
    print(list(df.columns))
    print("\nSample data:")
    print(df.head())

def main():
    try:
        # Process the Halide programs
        df = process_halide_programs()
        
        if df.empty:
            print("No data processed. Check your synthetic_data folder structure.")
            return
        
        # Print summary
        print_dataset_summary(df)
        
        # Save to CSV
        output_file = "halide_execution_dataset.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDataset saved to '{output_file}'")
        
        # Basic statistical analysis
        print("\nExecution time statistics:")
        print(df['execution_time_ms'].describe())
        
        # Correlation with execution time
        correlations = df.select_dtypes(include=[np.number]).corr()['execution_time_ms'].sort_values(ascending=False)
        print("\nTop 10 features correlated with execution time:")
        print(correlations.head(10))
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")

if __name__ == "__main__":
    main()
