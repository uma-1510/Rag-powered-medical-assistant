import re
import pandas as pd 
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import json
import numpy as np


CSV_PATH = "data/medquad.csv"
INDEX_PATH = "data/faiss_index.index"
METADATA_PATH= "data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "unconsious",
    "severe bleeding",
    "vision loss",
    "head injury",
    "vomitings",
    "diaherra",
    "more than 5 days"
]

def detect_emergency(text: str):
    text_lower= text.lower()
    return any(word in text_lower for word in EMERGENCY_KEYWORDS)

def split_medical_sections(answer: str):
    """
    Docstring for split_medical_sections
    
    :param answer: Description
    :type answer: str
    Light heuristic section splitter.
    Keeps your dataset intact but extracts context.

    """
    sections = {
        "symptom": [],
        "cause": [],
        "treatment":[],
        "warning": []
    }
    
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    for s in sentences:
        s_lower = s.lower()

        if any(k in s_lower for k in ["symptom", "sign", "feel", "pain"]):
            sections["symptom"].append(s)

        elif any(k in s_lower for k in ["cause", "because", "due to"]):
            sections["cause"].append(s)

        elif any(k in s_lower for k in ["treat", "relief", "rest", "drink", "medicine"]):
            sections["treatment"].append(s)

        elif any(k in s_lower for k in ["doctor", "seek", "emergency", "serious"]):
            sections["warning"].append(s)

    return sections

def build_medical_chunks(df):
    """
    Converts MedQuAD rows into context-aware chunks.
    """

    texts = []
    metadata = []

    for idx, row in df.iterrows():

        question = str(row["question"])
        answer = str(row["answer"])

        sections = split_medical_sections(answer)

        for chunk_type, sentences in sections.items():

            if not sentences:
                continue

            chunk_text = question + " " + " ".join(sentences)

            texts.append(chunk_text)

            metadata.append({
                "condition_question": question,
                "chunk_type": chunk_type,
                "text": chunk_text,
                "source": "MedQuAD",
                "source_id": f"medquad_{idx}",
                "is_emergency": detect_emergency(chunk_text)
            })

    return texts, metadata


def build_faiss_index(embeddings):
    dim= embeddings.shape[1]
    index= faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, index_path,metadata_path,metadata):
    faiss.write_index(index,index_path)
    print("faiss index saved")
    with open(metadata_path,"wb") as f:
        pickle.dump(metadata,f)


if __name__=="__main__":
    
    print("Loading dataset")
    df=pd.read_csv(CSV_PATH)
    texts, combined_metadata = build_medical_chunks(df)

    print(f"Generated {len(texts)} medical chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Encoding passages...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = build_faiss_index(embeddings)

    save_index_and_metadata(
        index,
        INDEX_PATH,
        METADATA_PATH,
        combined_metadata
    )

    print(f"Metadata saved to {METADATA_PATH}")
    print("Embedding and indexing complete")



