# # src/retriever.py

# from sentence_transformers import SentenceTransformer, util
# from typing import List, Tuple
# import pandas as pd

# # Load the S-BioBERT model for reranking
# # (This is a Sentence-BERT-style model fine-tuned for biomedical similarity)
# biomedical_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# def rerank_chunks_with_biomedical_similarity(
#     query: str,
#     chunks_df: pd.DataFrame,
#     top_k: int = 5
# ) -> List[Tuple[str, float]]:
#     """
#     Rerank the given chunks using S-BioBERT-based semantic similarity.

#     Args:
#         query (str): User query
#         chunks_df (pd.DataFrame): Filtered chunks dataframe with 'chunk_text' column
#         top_k (int): Number of top chunks to return

#     Returns:
#         List[Tuple[str, float]]: List of top-k (chunk_text, similarity_score)
#     """

#     if chunks_df.empty or "chunk_text" not in chunks_df.columns:
#         return []

#     chunk_texts = chunks_df['chunk_text'].tolist()

#     # Embed query and chunk_texts
#     query_embedding = biomedical_model.encode(query, convert_to_tensor=True)
#     chunk_embeddings = biomedical_model.encode(chunk_texts, convert_to_tensor=True)

#     # Compute cosine similarity
#     similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings).squeeze().tolist()

#     # Combine with text and sort
#     ranked = sorted(zip(chunk_texts, similarities), key=lambda x: x[1], reverse=True)

#     return ranked[:top_k]


import os
import json
import numpy as np
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Regex to strip any '(… route)' suffix for drug name matching
_DRUG_ROUTE_PATTERN = re.compile(r"\s*\([^)]*route\)", flags=re.IGNORECASE)

@lru_cache(maxsize=4)
def load_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load the saved embeddings numpy array.
    """
    return np.load(embeddings_path)


@lru_cache(maxsize=4)
def load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """
    Load the metadata JSON which contains chunk_id, drug_name, section, subsection, and chunk_text.
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def clean_drug_name(name: str) -> str:
    """Remove any '(… route)' suffix and extra whitespace."""
    return _DRUG_ROUTE_PATTERN.sub("", name).strip()


def filter_chunks(
    metadata: List[Dict[str, Any]],
    embeddings: np.ndarray,
    drug_name: str,
    section: str
) -> (List[int], np.ndarray, List[Dict[str, Any]]):
    """
    Filter metadata and embeddings by matching cleaned drug_name substring (case-insensitive) and section.
    Returns:
      - indices: List[int]
      - filtered_embeddings: np.ndarray of shape (n_filtered, dim)
      - filtered_meta: List of metadata dicts for these indices
    """
    indices = []
    meta_filtered = []
    target = drug_name.lower()

    for idx, meta in enumerate(metadata):
        cleaned = clean_drug_name(meta['drug_name']).lower()
        # substring match to include cases like 'ibuprofen' matching 'ibuprofen lysine'
        if target in cleaned and meta['section'] == section:
            indices.append(idx)
            meta_filtered.append(meta)

    if not indices:
        return [], np.empty((0, embeddings.shape[1])), []

    emb_filtered = embeddings[indices]
    return indices, emb_filtered, meta_filtered


def rerank_chunks(
    query: str,
    chunk_texts: List[str],
    model: SentenceTransformer,
    top_k: int = 5
) -> List[int]:
    """
    Given the user query and a list of chunk_texts, compute embeddings with `model` and return
    the indices of the top_k most similar chunks (by cosine similarity).
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    c_emb = model.encode(chunk_texts, convert_to_numpy=True)

    sims = cosine_similarity(q_emb, c_emb)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return top_idxs.tolist()


@lru_cache(maxsize=2)
def _get_reranker(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def retrieve(
    query: str,
    drug_name: str,
    section: str,
    embeddings_path: str = 'embeddings/drug_embeddings.npy',
    metadata_path: str = 'embeddings/drug_chunks_metadata.json',
    reranker_model_name: str = 'all-MiniLM-L6-v2',
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Load embeddings + metadata
    2. Filter for matching drug_name & section
    3. Rerank filtered chunks with a BioBERT-based sentence embedder
    4. Return top_k metadata entries with added similarity score

    Returns a list of metadata dicts enriched with 'score'.
    """
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    indices, emb_filtered, meta_filtered = filter_chunks(metadata, embeddings, drug_name, section)
    if not indices:
        print(f"No chunks found for drug='{drug_name}' and section='{section}'.")
        return []

    reranker = _get_reranker(reranker_model_name)
    chunk_texts = [m['chunk_text'] for m in meta_filtered]

    top_local_idxs = rerank_chunks(query, chunk_texts, reranker, top_k)

    # embed query once for scoring
    q_emb = reranker.encode([query], convert_to_numpy=True)

    results = []
    for local_idx in top_local_idxs:
        global_idx = indices[local_idx]
        text = chunk_texts[local_idx]
        c_emb = reranker.encode([text], convert_to_numpy=True)
        score = float(cosine_similarity(q_emb, c_emb)[0][0])
        entry = meta_filtered[local_idx].copy()
        entry['score'] = score
        results.append(entry)

    return results


# ── Smoke test ──
if __name__ == '__main__':
    query = "What are the side effects of Ibuprofen?"
    drug = "Ibuprofen"
    section = "Side Effects"

    top_chunks = retrieve(query, drug, section)
    if not top_chunks:
        exit(0)
    for chun in top_chunks:
        print(f"[{chun['score']:.3f}] {chun['chunk_text'][:100]}...")



