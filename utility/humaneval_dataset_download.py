from datasets import load_dataset
import pandas as pd

class HuggingFaceDBDownload:
    def download_humaneval():
        ds = load_dataset("openai/openai_humaneval")
        ds["test"].to_csv("humaneval_test.csv", index=False)

        # ds["train"].to_csv("humaneval_train.csv", index=False)

    def download_bigcodebench():
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
        pd.DataFrame(ds).to_csv("bigcodebench_test.csv", index=False)

    def download_cruxeval():
        ds = load_dataset("cruxeval-org/cruxeval")
        ds["test"].to_csv("cruxeval_test.csv")


if __name__ == "__main__":
    HuggingFaceDBDownload.download_cruxeval()