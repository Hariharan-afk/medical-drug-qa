# src/embedder.py

import os
import json
import argparse
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the flattened drug dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {'chunk_id', 'drug_name', 'section', 'subsection', 'chunk_text'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")
    return df

def embed_chunks(texts: list[str], model_name: str) -> np.ndarray:
    """Embed a list of texts using the given SentenceTransformer model."""
    print(f"Loading embedding model '{model_name}'…")
    model = SentenceTransformer(model_name)
    print("Embedding texts…")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings, axis=1)  # L2 normalize for cosine similarity
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index (cosine similarity via inner product on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_outputs(output_dir: str, index: faiss.Index, embeddings: np.ndarray, metadata: list[dict]):
    """Save the FAISS index, embedding matrix, and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.faiss"))
    np.save(os.path.join(output_dir, "drug_embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "drug_chunks_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved index, embeddings, and metadata to '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(
        description="Embed chunk_text with MiniLM-v6, build a FAISS index, and save outputs."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/flattened_drug_dataset_cleaned.csv",
        help="Path to the flattened drug CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Directory to save the FAISS index, embeddings, and metadata."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model to use (MiniLM-v6 checkpoint)."
    )
    args = parser.parse_args()

    print("Starting embedding pipeline…")
    df = load_data(args.csv_path)

    texts = df["chunk_text"].tolist()
    embeddings = embed_chunks(texts, args.model_name)

    index = build_faiss_index(embeddings)

    metadata = df[["chunk_id", "drug_name", "section", "subsection", "chunk_text"]].to_dict(orient="records")
    save_outputs(args.output_dir, index, embeddings, metadata)

    print("Embedding pipeline complete.")

if __name__ == "__main__":
    main()
