import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.query_normalizer import normalize_query
from utils.update_metadata import build_medquad_metadata


INDEX_PATH = "data/faiss_index.index"
METADATA_PATH = "data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer(EMBED_MODEL)

def compute_confidence(results):
    """
    Converts FAISS similarity score into confidence level.
    Higher score = better semantic match.
    """

    if not results:
        return 0.0

    # top result matters most
    top_score = results[0]["score"]

    # MiniLM cosine similarity usually ranges ~0.3–0.9
    confidence = float(top_score)

    return confidence

#  DIRECT ANSWER CHECK
def has_direct_answer(results, threshold=0.80):
    """
    If retrieval confidence is very high,
    we trust MedQuAD directly.
    """

    if not results:
        return False

    return results[0]["score"] >= threshold


def retrieve(query, k=5):
    # normalize query
    normalized_query = normalize_query(query)

    print(f"Original Query: {query}")
    print(f"Normalized Query: {normalized_query}")

    query_emb = model.encode(
        [normalized_query],
        normalize_embeddings=True
    )

    D, I = index.search(np.array(query_emb, dtype=np.float32), k)

    results = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx].copy()
        item['score'] = float(score)
        results.append(item)

    return results
