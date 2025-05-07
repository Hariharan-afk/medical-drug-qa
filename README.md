
# 💊 Medical Drug QA System

A retrieval-based question-answering system for medical drugs that combines intent classification, semantic chunk retrieval, and generative answer formulation. Designed to assist users in querying drug-related information efficiently using a structured and AI-powered pipeline.

---

## 🧠 Project Overview

This system enables users to ask natural language questions about medical drugs. The workflow includes:

1. **Web Scraping**: Drug-related data was scraped from the Mayo Clinic and flattened into a CSV format.
2. **Intent Classification**: The query is classified to identify:
   - The drug being referred to
   - The section of the drug (e.g., Description, Precautions, Dosage)
3. **Retrieval & Similarity**: Using `MiniLM-v6`, the system searches the most relevant chunks based on semantic similarity.
4. **Answer Generation**: The top 5 chunks are passed to a generative model to formulate a final, human-readable response.

---

## 📁 Folder Structure

```
medical-drug-qa/
├── app/                    # Streamlit UI and chat utilities
│   ├── chat_utils.py
│   └── main.py
├── src/                    # Core backend modules
│   ├── answer_generator.py
│   ├── chatbot.py
│   ├── config.py
│   ├── drug_dictionary.py
│   ├── embedder.py
│   ├── intent_classifier.py
│   └── retriever.py
├── download_assets.py      # Script to download large data/model files
├── requirements.txt        # Python dependencies
```

---

## 🔽 Download Required Data

Due to GitHub's file size limits, key assets (embeddings and datasets) are hosted on Google Drive. Run the following command to automatically download them:

```bash
python download_assets.py
```

This will download:
- `embeddings/faiss_index.faiss`
- `embeddings/drug_embeddings.npy`
- `embeddings/drug_chunks_metadata.json`
- `data/flattened_drug_dataset_cleaned.csv`

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Hariharan-afk/medical-drug-qa.git
   cd medical-drug-qa
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate       # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the embeddings and data:
   ```bash
   python download_assets.py
   ```

5. Launch the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```

---

## 🛠️ Technologies Used

- **Python**
- **MiniLM-v6** (Sentence Transformers)
- **FAISS** (Similarity Search)
- **HuggingFace Transformers**
- **Streamlit**
- **Pandas / NumPy / Scikit-learn**

---

## ✨ Future Improvements

- Use LLM APIs (e.g., Groq or OpenAI) for advanced answer generation
- Improve chunk reranking with BGE or S-BioBERT embeddings
- Add unit tests and Streamlit Cloud deployment

---

## 📬 Contact

Project developed by Hariharan Chandrasekar.  
If you'd like to collaborate or explore this further, feel free to connect!
