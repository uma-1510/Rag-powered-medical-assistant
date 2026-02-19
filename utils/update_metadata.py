import pandas as pd
import pickle

CSV_PATH = "data/medquad.csv"
METADATA_PATH = "data/metadata.pkl"


def build_medquad_metadata():

    df = pd.read_csv(CSV_PATH)

    metadata = []

    for _, row in df.iterrows():

        combined_text = (
            str(row["question"]) +
            " [SEP] " +
            str(row["answer"])
        )

        metadata.append({
            "text": combined_text,
            "question": row["question"],
            "answer": row["answer"],
            "source": row.get("source", "MedQuAD"),
            "focus_area": row.get("focus_area", ""),
            # simple safety heuristic
            "is_emergency": any(word in str(row["answer"]).lower()
                                for word in ["blindness", "vision loss",
                                             "severe", "emergency"])
        })

    return metadata


if __name__ == "__main__":

    metadata = build_medquad_metadata()

    print(f"MedQuAD entries: {len(metadata)}")

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Metadata saved at {METADATA_PATH}")
