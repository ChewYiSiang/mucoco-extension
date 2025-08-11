import json
import pandas as pd
from typing import Dict

def convert_json(file_path: str):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line.strip()
            if len(line) > 0:
                data_entry : Dict = json.loads(line)
                data.append(data_entry)

    df = pd.DataFrame(data)
    new_file_name = Path(file_path).stem + ".csv"
    csv_file_path = Path(file_path).parent / new_file_name
    df.to_csv(csv_file_path)
    print(f'Created {new_file_name}.')

if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os

    parent_dir = Path(__file__).parent.parent
    sys.path.append(str(parent_dir))
    
    file_path = parent_dir / "datasets" / "open_ended_format" / "CodeSense"
    jsonl_file_names = [name for name in os.listdir(file_path) if name.endswith('jsonl')]

    idx = 0
    while idx < len(jsonl_file_names):
        jsonl_file_name = jsonl_file_names[idx]
        csv_file_name = Path(jsonl_file_name).stem + ".csv"
        if csv_file_name in os.listdir(file_path):
            jsonl_file_names.pop(idx)
            print(f"{csv_file_name} already exists in the directory. If you wish to create a new csv file, delete the old existing entry.")

    for json_file_name in jsonl_file_names:
        json_file_path = os.path.join(file_path, json_file_name)
        convert_json(json_file_path)