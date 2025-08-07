import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

INDEX_PATH = "data/faiss_index.index"
METADATA_PATH = "data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


_index = None
_metadata = None
_model = None

# def get_index_metadata_model():
#     global _index, _metadata, _model
#     if _index is None or _metadata is None:
#         print("Loading FAISS index and metadata...")
#         _index = faiss.read_index(INDEX_PATH)
#         with open(METADATA_PATH, "rb") as f:
#             _metadata = pickle.load(f)
#     if _model is None:
#         _model = SentenceTransformer(EMBED_MODEL)
#     return _index, _metadata, _model


# Load FAISS index and metadata
def load_index_and_metadata():
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Retrieve top-k passages for a query
def retrieve(query, k=5):
    index, metadata = load_index_and_metadata ()
    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb, dtype=np.float32), k)
    results = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx].copy()
        item['score'] = float(score)
        results.append(item)
    return results

from sentence_transformers import CrossEncoder

def rerank_cross_encoder(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, item['answer']) for item in candidates]
    scores = cross_encoder.predict(pairs)
    for item, score in zip(candidates, scores):
        item['rerank_score'] = float(score)
    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    return candidates


def print_retrieved_oneliners(results, max_items=3, maxlen=80):
    """
    Print retrieved Q&A pairs as one-liners, truncating question and answer for readability.
    """
    print(f"\nRetrieved Documents (top {min(max_items, len(results))}):")
    for i, item in enumerate(results[:max_items], 1):
        q = item['question'].replace('\n', ' ').strip()
        a = item['answer'].replace('\n', ' ').strip()
        q = (q[:maxlen] + '...') if len(q) > maxlen else q
        a = (a[:maxlen] + '...') if len(a) > maxlen else a
        print(f"{i}. Q: {q}")
        print(f"   A: {a}")

    print()  # for newline

def generate_alternative_queries(original_query, llm_generate_fn, n=3):
    """
    Generate multiple alternative query reformulations using an LLM.
    `llm_generate_fn` is a user-provided function that, given a prompt, returns a list of strings.
    
    For example, it can be a wrapper around your Gemini API to get reformulations.
    """
    prompt = (
        f"Generate {n} different questions that ask the same as: \"{original_query}\" "
        "but from different perspectives or using different wording."
    )
    alternative_queries = llm_generate_fn(prompt)
    # Optionally ensure the original query is included (de-duplication etc)
    if original_query not in alternative_queries:
        alternative_queries.insert(0, original_query)
    return alternative_queries


def reciprocal_rank_fusion(all_results, k_smooth=60):
    """
    Fuse multiple retrieval result lists using Reciprocal Rank Fusion (RRF).
    all_results: list of (query, result_list) tuples.
    Each result_list is a list of dicts with a unique identifier key (e.g., 'document_id').

    Returns a fused list of documents sorted by combined RRF score.
    """
    doc_scores = defaultdict(lambda: {'score': 0.0, 'doc': None})

    for _, results in all_results:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get('document_id') or doc.get('qapair_pid') or str(hash(doc['question'] + doc['answer']))
            score = 1.0 / (rank + k_smooth)
            if doc_scores[doc_id]['doc'] is None:
                doc_scores[doc_id]['doc'] = doc.copy()
            doc_scores[doc_id]['score'] += score

    # Sort docs by descending RRF score
    fused = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in fused]


def rag_fusion_search(query, llm_generate_fn, k_per_query=5, top_k=10):
    """
    The RAG Fusion search pipeline:
    - Generate multiple query reformulations
    - Retrieve documents per reformulated query
    - Fuse retrieval results using RRF
    - Optionally rerank with cross-encoder
    - Return final reranked candidates truncated to top_k
    """
    # Step 1: Generate alternative queries
    alternative_queries = generate_alternative_queries(query, llm_generate_fn, n=3)
    
    # Step 2: Retrieve results for each alternative query
    all_results = []
    for alt_q in alternative_queries:
        results = retrieve(alt_q, k=k_per_query)
        all_results.append((alt_q, results))
        print(all_results)
    
    # Step 3: Fuse results using RRF
    fused_results = reciprocal_rank_fusion(all_results, k_smooth=60)
    
    # Step 4: Rerank fused results with cross-encoder on *original* query
    reranked_results = rerank_cross_encoder(query, fused_results)
    
    # Step 5: Return top final results
    return reranked_results[:top_k]