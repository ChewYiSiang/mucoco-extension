import pandas as pd 
import sys
import os
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utility.data_log_functions import DataLogHelper

# Reference file (the one to compare all others against)
reference_path = "results/Llama/Input_Prediction/FEW_SHOT/Llama-3.1-8B-Instruct_few_shot_no_mutation2.csv"

# Directory containing files to compare
comparison_dir = "results/Llama/Input_Prediction"

print("=== LLAMA INPUT PREDICTION COMPARISON ===")
print(f"Reference file: {reference_path}")
print(f"Comparing all CSV files in: {comparison_dir}")
print("=" * 60)

# Load reference file
try:
    reference_df = pd.read_csv(reference_path)
    print(f"\nReference file shape: {reference_df.shape}")
except FileNotFoundError:
    print(f"Error: Reference file not found at {reference_path}")
    sys.exit(1)

# Find all CSV files in the directory and subdirectories
csv_files = []
for root, dirs, files in os.walk(comparison_dir):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Remove reference file from comparison list
csv_files = [f for f in csv_files if f != reference_path]

print(f"\nFound {len(csv_files)} CSV files to compare:")
for file in sorted(csv_files):
    print(f"  - {file}")

print("\n" + "=" * 60)
print("COMPARISON RESULTS:")
print("=" * 60)

# Compare each file with the reference
for file_path in sorted(csv_files):
    try:
        print(f"\nComparing: {os.path.basename(file_path)}")
        print(f"Full path: {file_path}")
        
        # Load comparison file
        comparison_df = pd.read_csv(file_path)
        print(f"File shape: {comparison_df.shape}")
        
        # Perform comparison
        ref_inconsistencies, comp_inconsistencies = DataLogHelper.compare_code_generation_dataframe_results(
            log1=reference_df, 
            log2=comparison_df
        )
        
        print(f"Reference inconsistencies: {ref_inconsistencies}")
        print(f"Comparison file inconsistencies: {comp_inconsistencies}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print("-" * 40)

print("\nComparison complete!")