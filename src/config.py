# src/config.py

CSV_PATH = "data/flattened_drug_dataset_cleaned.csv"
TOP_K = 20
# Use a currently supported model by default. You can override with the GROQ_MODEL
# environment variable (or in a .env file) if you prefer a different model.
GROQ_MODEL = "openai/gpt-oss-120b"
