from datasets import load_dataset

# 1) Load the dataset
ds = load_dataset("openai/openai_humaneval")

# 2) Write the TEST split directly to CSV
ds["test"].to_csv("humaneval_test.csv", index=False)

# (Optionally do the same for train/validation if they exist)
# ds["train"].to_csv("humaneval_train.csv", index=False)
