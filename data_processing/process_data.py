import os
import pandas as pd
import json

def load_dataset():
    # Read the raw CSV from the data/raw folder.
    data_path = os.path.join("data", "raw", "casters_cards.csv")
    df = pd.read_csv(data_path, dtype={'level': 'str', 'attack': 'str', 'health': 'str'}).fillna("")
    # Convert to list of dictionaries.
    dataset = json.loads(df.to_json(orient="records"))
    return dataset

def save_processed_dataset(dataset, output_file=os.path.join("data", "processed", "casters_cards.jsonl")):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    dataset = load_dataset()
    save_processed_dataset(dataset)
    print("Dataset processed and saved to data/processed/casters_cards.jsonl")
