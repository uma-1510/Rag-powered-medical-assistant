import pandas as pd 
import faiss
import pickle
from sentence_transformers import SentenceTransformer


CSV_PATH = "data/medquad.csv"
INDEX_PATH = "data/faiss_index.index"
METADATA_PATH= "data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

def build_faiss_index(embeddings):
    dim= embeddings.shape[1]
    index= faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, index_path,metadata_path,df):
    faiss.write_index(index,index_path)
    print("faiss index saved")
    metadata = df.to_dict(orient="records")
    with open(metadata_path,"wb") as f:
        pickle.dump(metadata,f)


if __name__=="__main__":

    df=pd.read_csv(CSV_PATH)

    texts= (df["question"].astype(str) + "[SEP]" + df["answer"].astype(str)).tolist()

    print("loading model")
    model= SentenceTransformer(EMBED_MODEL)
    print(f"Encoding {len(texts)} passages...")
    embeddings= model.encode(
        texts,
        progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    index= build_faiss_index(embeddings)
    save_index_and_metadata(index,INDEX_PATH,METADATA_PATH,df)
    print(f"Metadata saved to {METADATA_PATH}")

    print("Embedding and indexing complete")


