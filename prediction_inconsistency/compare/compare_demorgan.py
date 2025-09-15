import pandas as pd 
from utility.data_log_functions import DataLogHelper

log1_path = "results/code_inconsistencies/mistral/Final_DeMorgan_Mutation/mistral-small-2506_zero_shot_demorgan_mutation.csv"
log2_path = "results/code_inconsistencies/mistral/Final_DeMorgan_Mutation/mistral-small-2506_zero_shot_No_Mutation_DeMorgan_Filtered.csv"

log1 = pd.read_csv(log1_path)
log2 = pd.read_csv(log2_path)

# compare results
log1_inconsistencies, log2_inconsistencies = DataLogHelper.compare_code_generation_dataframe_results(log1=log1, log2=log2)

print(f"Mutation inconsistencies: {log1_inconsistencies}")
print(f"Mutation inconsistencies: {log2_inconsistencies}")