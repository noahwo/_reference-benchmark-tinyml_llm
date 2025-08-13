#!/usr/bin/env python3
"""
Script to combine all CSV files from data_analysis/2025/*/processed_data/**/*.csv
into a single CSV file categorized by batch_id column.
"""
# %%
import glob
import os
from pathlib import Path

import pandas as pd

# %%``

def combine_csv_files():
    # %%
    # Base directory for data analysis
    base_dir = "/Users/hann/Projects/reference-benchmark-tinyml_llm/data_analysis/2025"
    
    # Pattern to find all CSV files
    # The folder name need to be something greater than 05 as the leading two numbers separated by dot. 
    
    
    
    # List all folders in base_dir and filter those with leading numbers > 05
    valid_folders = []
    for folder in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, folder)):
            parts = folder.split('.')
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) >= 4:
                valid_folders.append(folder)

    # Build glob patterns for each valid folder
    csv_files = []
    for folder in valid_folders:
        pattern = os.path.join(base_dir, folder, "processed_data", "**", "*.csv")
        csv_files.extend(glob.glob(pattern, recursive=True))
    
    # Find all CSV files
    # csv_files = glob.glob(csv_pattern, recursive=True)
    print(f"Valid folders found: {valid_folders}")
    print(f"Found {len(csv_files)} CSV files to combine")
    
    
    # %%
    # List to store all dataframes
    all_dataframes = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add metadata columns for tracking
            df['source_file'] = os.path.basename(csv_file)
            df['source_path'] = csv_file
            
            # Extract date from path (e.g., "07.29" from path)
            path_parts = Path(csv_file).parts
            date_index = next((i for i, part in enumerate(path_parts) if part == "2025"), None)
            if date_index and date_index + 1 < len(path_parts):
                df['test_date'] = path_parts[date_index + 1]
            else:
                df['test_date'] = 'unknown'
            
            # Extract model info from the processed_data folder name
            processed_data_index = next((i for i, part in enumerate(path_parts) if part == "processed_data"), None)
            if processed_data_index and processed_data_index + 1 < len(path_parts):
                df['model_config'] = path_parts[processed_data_index + 1]
            else:
                df['model_config'] = 'unknown'
            
            all_dataframes.append(df)
            print(f"Processed: {csv_file} - {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("No CSV files were successfully processed!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df['generation_count'] = combined_df['generation_count'].astype(int)
    # Sort by batch_id and then by num_run if available
    if 'batch_id' in combined_df.columns:
        sort_columns = ['batch_id']
        if 'num_run' in combined_df.columns:
            sort_columns.append('num_run')
        combined_df = combined_df.sort_values(sort_columns)
    
    # Output file path
    output_file = "/Users/hann/Projects/reference-benchmark-tinyml_llm/combined_tinyml_benchmark_data.csv"
    
    # Save combined CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined data saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total unique batch_ids: {combined_df['batch_id'].nunique()}")
    
    # Display summary statistics
    print("\nBatch ID Summary:")
    batch_summary = combined_df['batch_id'].value_counts().head(20)
    print(batch_summary)
    
    print("\nTest Date Summary:")
    date_summary = combined_df['test_date'].value_counts()
    print(date_summary)
    
    print("\nModel Configuration Summary:")
    model_summary = combined_df['model_config'].value_counts()
    print(model_summary)
    
    return output_file

if __name__ == "__main__":
    combine_csv_files()
 
