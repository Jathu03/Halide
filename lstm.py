import os
import pandas as pd
from pathlib import Path

def process_halide_programs(root_dir="synthetic_data"):
    """
    Process Halide programs from synthetic_data folder and create a dataset representation.
    
    Args:
        root_dir (str): Root directory containing Halide program subfolders
        
    Returns:
        dict: Dataset containing program information
        DataFrame: Structured representation of the programs and schedules
    """
    
    # Initialize data structures
    dataset = {
        "programs": {},
        "schedules": {},
        "file_paths": {}
    }
    
    # List to store data for DataFrame
    data_records = []
    
    # Ensure root directory exists
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory {root_dir} not found")
    
    # Iterate through all subfolders (each representing a program)
    for program_folder in root_path.iterdir():
        if program_folder.is_dir():
            program_name = program_folder.name
            dataset["programs"][program_name] = []
            
            # Process each schedule file in the program folder
            for schedule_file in program_folder.iterdir():
                if schedule_file.is_file():
                    schedule_name = schedule_file.stem  # filename without extension
                    
                    # Store file path
                    dataset["file_paths"][f"{program_name}/{schedule_name}"] = str(schedule_file)
                    
                    # Add schedule to program's list
                    dataset["programs"][program_name].append(schedule_name)
                    
                    # Store schedule info
                    dataset["schedules"][f"{program_name}/{schedule_name}"] = {
                        "program": program_name,
                        "schedule": schedule_name,
                        "path": str(schedule_file),
                        "extension": schedule_file.suffix
                    }
                    
                    # Add record for DataFrame
                    data_records.append({
                        "program_name": program_name,
                        "schedule_name": schedule_name,
                        "file_path": str(schedule_file),
                        "file_extension": schedule_file.suffix,
                        "last_modified": schedule_file.stat().st_mtime
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    return dataset, df

def print_dataset_summary(dataset, df):
    """
    Print a summary of the processed dataset.
    
    Args:
        dataset (dict): Processed dataset dictionary
        df (DataFrame): DataFrame representation
    """
    print(f"Total programs found: {len(dataset['programs'])}")
    print(f"Total schedules found: {len(dataset['schedules'])}")
    print("\nPrograms and their schedule counts:")
    for program, schedules in dataset['programs'].items():
        print(f"- {program}: {len(schedules)} schedules")
    print("\nDataset DataFrame shape:", df.shape)
    print("\nDataFrame columns:", list(df.columns))

def main():
    try:
        # Process the Halide programs
        dataset, df = process_halide_programs()
        
        # Print summary
        print_dataset_summary(dataset, df)
        
        # Example usage of the data
        print("\nSample of the DataFrame:")
        print(df.head())
        
        # Save the DataFrame to CSV (optional)
        df.to_csv("halide_programs_dataset.csv", index=False)
        print("\nDataset saved to 'halide_programs_dataset.csv'")
        
        return dataset, df
        
    except Exception as e:
        print(f"Error processing Halide programs: {str(e)}")
        return None, None

if __name__ == "__main__":
    dataset, df = main()
