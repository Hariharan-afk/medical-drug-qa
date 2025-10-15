import os
import sys
import re
import streamlit as st

# Ensure 'src' folder is on the PYTHONPATH for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from intent_classifier import classify_intent
from retriever import retrieve
from answer_generator import generate_answer
from chat_utils import append_to_history, render_chat_history, auto_scroll_to_bottom


def ensure_assets():
    """Ensure required data and embedding files exist; auto-download if missing."""
    required = [
        "data/flattened_drug_dataset_cleaned.csv",
        "embeddings/faiss_index.faiss",
        "embeddings/drug_embeddings.npy",
        "embeddings/drug_chunks_metadata.json",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        with st.spinner("Preparing assets (first-time setup)..."):
            try:
                import download_assets  # noqa: F401
            except Exception as e:
                st.error(f"Failed to prepare assets automatically. Missing: {missing}")
                st.exception(e)


ensure_assets()

st.set_page_config(page_title="Medical Drug QA", layout="wide")
st.title("💊 Medical Drug Question Answering System")

# Initialize chat history and context
if "history" not in st.session_state:
    st.session_state.history = []
if "current_drug" not in st.session_state:
    st.session_state.current_drug = None

# Input box
query = st.text_input(
    "Ask a medical drug question:",
    placeholder="e.g., What are the side effects of Ibuprofen?"
)

if query:
    st.markdown("---")
    # Self-introduction handling
    if re.search(r"\b(who (are you|r u)|what are you|tell me about yourself)\b", query, flags=re.IGNORECASE):
        intro = (
            "I am a Medical Drug Question Answering System. "
            "I can help you find information on approved medications, their uses, dosages, side effects, and more. "
            "Just ask me about a drug, and I'll retrieve relevant details from trusted drug documentation."
        )
        st.success(intro)
        append_to_history(query, {"section": "Introduction"}, [], intro)
    else:
        # 1) Intent classification
        with st.spinner("🔎 Detecting intent..."):
            intent = classify_intent(query)
            drug = intent.get("drug_name")
            section = intent.get("section")

            # Contextual drug resolution: remember or reuse
            if drug:
                st.session_state.current_drug = drug
            else:
                if st.session_state.current_drug:
                    drug = st.session_state.current_drug

        if not drug:
            st.warning("Sorry, I don’t have information on that drug in my database.")
        else:
            # 2) Retrieval
            with st.spinner("📚 Retrieving relevant chunks..."):
                chunks = retrieve(query, drug, section)
            if not chunks:
                st.warning(f"No '{section}' information found for {drug}.")
            else:
                # 3) Answer generation
                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(query, chunks)

                # Save to history
                append_to_history(query, intent, chunks, answer)

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs([
                    "✅ Final Answer", "🧭 Intent & Metadata", "📄 Top Results"
                ])

                with tab1:
                    st.markdown(f"### Final Answer for **{drug}** ({section})")
                    st.success(answer)

                with tab2:
                    st.markdown("### Detected Intent")
                    st.json(intent)

                with tab3:
                    st.markdown(f"### Top {len(chunks)} Results (by relevance)")
                    for chunk in chunks:
                        text = chunk.get('chunk_text', '')
                        st.markdown(f"- {text}")

# Show chat history at bottom
render_chat_history()
auto_scroll_to_bottom()

