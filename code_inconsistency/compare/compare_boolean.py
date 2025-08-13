import pandas as pd 
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utility.data_log_functions import DataLogHelper

# Fixed paths to match actual file locations
log1_path = "results/code_inconsistencies/mistral/Final_Boolean/mistral-small-2506_zero_shot_boolean_literal_baseline.csv"
#results/code_inconsistencies/mistral/Final_Boolean/mistral-small-2506_zero_shot_boolean_literal_baseline.csv
log2_path = "results/code_inconsistencies/mistral/Final_Boolean/mistral-small-2506_zero_shot_boolean_literal_mutation.csv"

print("=== BOOLEAN LITERAL COMPARISON ===")
print(f"Log1 (Baseline): {log1_path}")
print(f"Log2 (Mutation): {log2_path}")

log1 = pd.read_csv(log1_path)
log2 = pd.read_csv(log2_path)

print(f"\nLog1 (Baseline) shape: {log1.shape}")
print(f"Log2 (Mutation) shape: {log2.shape}")

# compare results
log1_inconsistencies, log2_inconsistencies = DataLogHelper.compare_code_generation_dataframe_results(log1=log1, log2=log2)

print(f"\nBaseline inconsistencies: {log1_inconsistencies}")
print(f"Mutation inconsistencies: {log2_inconsistencies}")