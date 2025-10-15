# Medical Drug QA

Retrieval-augmented question answering for medical drugs with a Streamlit UI. It classifies intent (drug + section), retrieves semantically relevant chunks, and generates a grounded answer using an LLM.

---

## Overview

- Ingestion: Drug documentation is flattened into `data/flattened_drug_dataset_cleaned.csv` and chunked with metadata.
- Embeddings: Sentence embeddings built with `all-MiniLM-L6-v2` and stored alongside a FAISS index in `embeddings/`.
- Intent: `src/intent_classifier.py` extracts the drug (via dictionary lookup) and classifies a section via semantic similarity.
- Retrieval: `src/retriever.py` filters by drug + section, then ranks relevant chunks.
- Generation: `src/answer_generator.py` composes a grounded response (Groq API).

---

## Folder Structure

```
medical-drug-qa/
├─ app/                      # Streamlit UI
│  ├─ main.py
│  └─ chat_utils.py
├─ src/                      # Core logic
│  ├─ answer_generator.py
│  ├─ chatbot.py             # Optional CLI
│  ├─ config.py
│  ├─ drug_dictionary.py
│  ├─ embedder.py
│  ├─ intent_classifier.py
│  └─ retriever.py
├─ embeddings/               # Downloaded at runtime (gitignored)
├─ data/                     # Downloaded at runtime (gitignored)
├─ app.py                    # HF Spaces entrypoint for Streamlit
├─ download_assets.py        # Downloads CSV + embeddings
├─ requirements.txt
└─ .gitignore
```

---

## Setup (Local)

1) Create environment and install deps

```
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2) Download dataset + embeddings

```
python download_assets.py
```

3) Configure environment

Create a `.env` file with:

```
GROQ_API_KEY=your_key
# Optional
GROQ_MODEL=openai/gpt-oss-120b
```

4) Run

```
streamlit run app/main.py
```

---

## Deploy: Hugging Face Spaces (Streamlit)

1) Push this repo to GitHub (ensure `data/` and `embeddings/` are not committed; they are downloaded at runtime).

2) Create a Space
- Go to huggingface.co/spaces → New Space
- SDK: Streamlit
- Hardware: CPU Basic
- Connect your GitHub repo or upload files

3) App entrypoint
- This repo includes `app.py` at the root which imports the Streamlit app, so Spaces will run it automatically.

4) Set secrets/variables
- In the Space: Settings → Variables and secrets
  - `GROQ_API_KEY`: your API key
  - Optional `GROQ_MODEL`

5) First build/run
- The app installs requirements and then pulls assets via `download_assets.py` if missing.

---

## Notes

- If you want to rebuild embeddings: run `src/embedder.py` to refresh `embeddings/` from the CSV.
- `src/retriever.py` and `src/intent_classifier.py` use MiniLM embeddings for speed; you can swap in a biomedical reranker for higher accuracy.
- Large artifacts are gitignored; the app auto-downloads them on first run.

