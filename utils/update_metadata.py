import pandas as pd
import json
import pickle

CSV_PATH = "data/medquad.csv"
JSON_PATH = "data/bench_mark.json"
METADATA_PATH = "data/metadata.pkl"

def load_json_data(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    medqa_data = data.get("medqa", {})
    for key, entry in medqa_data.items():
        question = entry.get("question", "")
        options = entry.get("options", {})
        answer_key = entry.get("answer", "")
        answer_text = options.get(answer_key, "") if options else ""
        records.append({"question": question, "answer": answer_text})
    return records

if __name__ == "__main__":
    # Load MedQuAD csv metadata
    df = pd.read_csv(CSV_PATH)
    medquad_metadata = df.to_dict(orient="records")

    # Load benchmark.json metadata
    benchmark_metadata = load_json_data(JSON_PATH)

    # Add a source field to distinguish datasets, optional but recommended
    for item in medquad_metadata:
        item["source"] = "medquad"

    for item in benchmark_metadata:
        item["source"] = "benchmark"

    # Combine metadata lists
    combined_metadata = medquad_metadata + benchmark_metadata

    print(f"MedQuAD entries: {len(medquad_metadata)}")
    print(f"Benchmark entries: {len(benchmark_metadata)}")
    print(f"Combined metadata entries: {len(combined_metadata)}")

    # Save combined metadata pickle
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(combined_metadata, f)

    print(f"Updated metadata saved at {METADATA_PATH}")
