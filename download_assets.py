# download_assets.py

import gdown
import os

# Create necessary folders
os.makedirs("embeddings", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Google Drive File IDs
DATASET_ID = "13EjNIafiiewrbkaZggb-OF0g9FisQIlc"
FAISS_ID = "1atILOj3AQzhgqrTos0eHFyqWRhYgqF8J"
EMBEDDINGS_ID = "1q3gccCH8d9QpoknnlRSenMKoprViS1w-"
METADATA_ID = "1NpLxAAlctMe0AFh71-i9k8Vq7aj9DO6w"

# Download files
print("📥 Downloading flattened_drug_dataset_cleaned.csv ...")
gdown.download(f"https://drive.google.com/uc?id={DATASET_ID}", "data/flattened_drug_dataset_cleaned.csv", quiet=False)

print("📥 Downloading faiss_index.faiss ...")
gdown.download(f"https://drive.google.com/uc?id={FAISS_ID}", "embeddings/faiss_index.faiss", quiet=False)

print("📥 Downloading drug_embeddings.npy ...")
gdown.download(f"https://drive.google.com/uc?id={EMBEDDINGS_ID}", "embeddings/drug_embeddings.npy", quiet=False)

print("📥 Downloading drug_chunks_metadata.json ...")
gdown.download(f"https://drive.google.com/uc?id={METADATA_ID}", "embeddings/drug_chunks_metadata.json", quiet=False)

print("✅ All required files downloaded successfully.")
