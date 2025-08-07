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
JSON_PATH= "data/bench_mark.json"

def load_json_data(json_path:str):
    with open(json_path,"r") as f:
        data=json.load(f)
    records=[]
    medqa_data= data.get("medqa",{})
    for key,entry in medqa_data.items():
        question=entry.get("question", "")
        options=entry.get("options","")
        answer=entry.get("answer","")

        answer_text= options.get(answer,"") if options else ""

        records.append({"question": question, "answer": answer_text})
        return records

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

    df=pd.read_csv(CSV_PATH)

    results= load_json_data(JSON_PATH)

    # new_json_records = [r for r in results if r["question"] not in texts]
    # print(f"Filtered {len(new_json_records)} new JSON records to embed")

    texts= (df["question"].astype(str) + "[SEP]" + df["answer"].astype(str)).tolist()
    json_texts = [(rec["question"] + "[SEP]" + rec["answer"]) for rec in results]


    print("loading model")
    model= SentenceTransformer(EMBED_MODEL)
    print(f"Encoding {len(texts)} passages...")
    csv_embeddings= model.encode(
        texts,
        progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"Encoding {len(json_texts)} new JSON passages...")
    if len(json_texts) > 0:
        json_embeddings = model.encode(
            json_texts,
            progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    else:
        json_embeddings = np.empty((0, csv_embeddings.shape[1]), dtype=np.float32)
    
    combined_embeddings = np.concatenate([csv_embeddings, json_embeddings], axis=0)

    medquad_metadata = df.to_dict(orient="records")
    combined_metadata = medquad_metadata + json_texts

    
    index= build_faiss_index(combined_embeddings)
    save_index_and_metadata(index,INDEX_PATH,METADATA_PATH,combined_metadata)
    print(f"Metadata saved to {METADATA_PATH}")

    print("Embedding and indexing complete")


