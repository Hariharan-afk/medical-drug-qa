# import subprocess

# try:
#     import en_ner_bc5cdr_md
# except ImportError:
#     subprocess.run(
#         ["pip", "install", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz"],
#         check=True
#     )

# import sys
# import os
# import pandas as pd
# import streamlit as st

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.intent_classifier import extract_intents
# from src.filter_chunks import filter_chunks
# from src.retriever import rerank_chunks_with_biomedical_similarity
# from src.answer_generator import generate_answer
# from src.config import CSV_PATH, TOP_K, GROQ_MODEL
# from chat_utils import append_to_history, render_chat_history, auto_scroll_to_bottom

# st.set_page_config(page_title="Medical Drug QA", layout="wide")
# st.title("\U0001F48A Medical Drug Question Answering System")

# if "history" not in st.session_state:
#     st.session_state.history = []

# @st.cache_data
# def load_dataset():
#     df = pd.read_csv(CSV_PATH)
#     drug_names = df["drug_name"].dropna().unique().tolist()
#     section_names = df["section"].dropna().unique().tolist()
#     return df, drug_names, section_names

# with st.spinner("\U0001F4C5 Loading medical drug dataset..."):
#     df, drug_names, section_names = load_dataset()

# query = st.text_input("Ask a medical drug question:", placeholder="e.g., What is azithromycin used for?")

# def highlight_terms(text, terms):
#     for term in terms:
#         if isinstance(term, str) and term.lower() in text.lower():
#             text = text.replace(term, f"**{term}**")
#     return text

# if query:
#     st.markdown("---")
#     st.subheader("\U0001F504 Processing your query...")

#     try:
#         intents = extract_intents(query, drug_names, section_names)
#         if "section" in intents and "section_list" not in intents:
#             intents["section_list"] = [intents["section"]]

#         filtered_df = filter_chunks(df, intents)
#         if filtered_df.empty:
#             st.warning("\u26A0\uFE0F No chunks matched the intent. Try rephrasing your query.")
#             st.stop()

#         top_chunks = rerank_chunks_with_biomedical_similarity(query, filtered_df, top_k=TOP_K)
#         context_chunks = [chunk for chunk, _ in top_chunks]

#         with st.spinner("\U0001F9E0 Generating answer using Groq..."):
#             final_answer = generate_answer(query, context_chunks, model=GROQ_MODEL)

#         append_to_history(query, intents, top_chunks, final_answer)

#         tab1, tab2, tab3, tab4 = st.tabs(["\U0001F4AC Final Answer", "\U0001F50D Intents", "\U0001F4CA Filtered Chunks", "\U0001F9EC Top Chunks"])

#         with tab1:
#             st.markdown("### \U0001F9E0 Final Answer")
#             st.success(final_answer)

#         with tab2:
#             st.markdown("### Detected Intents")
#             st.json(intents)

#         with tab3:
#             display_cols = [col for col in ["drug_name", "section", "chunk", "text"] if col in filtered_df.columns]
#             st.write(f"Filtered to {len(filtered_df)} candidate chunks.")
#             st.dataframe(filtered_df[display_cols].head(10))

#         with tab4:
#             for i, (chunk, score) in enumerate(top_chunks):
#                 highlighted = highlight_terms(chunk, [query] + intents.get("drug_list", []) + intents.get("section_list", []))
#                 st.markdown(f"**Chunk {i+1}** (Score: {score:.4f})")
#                 st.markdown(highlighted[:500] + "...")

#     except Exception as e:
#         st.error("\u274C An error occurred while processing your request.")
#         st.exception(e)

# # === Display Chat History === #
# render_chat_history()
# auto_scroll_to_bottom()

# main.py
import sys
import os
import streamlit as st

# Ensure 'src' folder is on the PYTHONPATH for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from intent_classifier import classify_intent
from retriever import retrieve
from answer_generator import generate_answer
from chat_utils import append_to_history, render_chat_history, auto_scroll_to_bottom

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
    import re
    if re.search(r"\b(who (are you|r u)|what are you|tell me about yourself)\b", query, flags=re.IGNORECASE):
        intro = (
            "I am a Medical Drug Question Answering System. "
            "I can help you find information on approved medications, their uses, dosages, side effects, and more. "
            "Just ask me about a drug, and I'll retrieve relevant details from trusted drug documentation."
        )
        st.success(intro)
        append_to_history(query, {"section": "Introduction"}, [], intro)
    else:
        # 1️⃣ Intent classification
        with st.spinner("🔍 Detecting intent..."):
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
            st.warning("⚠️ Sorry, I don’t have information on that drug in my database.")
        else:
            # 2️⃣ Retrieval
            with st.spinner("📂 Retrieving relevant chunks..."):
                chunks = retrieve(query, drug, section)
            if not chunks:
                st.warning(f"⚠️ No '{section}' information found for {drug}.")
            else:
                # 3️⃣ Answer generation
                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(query, chunks)

                # Save to history
                append_to_history(query, intent, chunks, answer)

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs([
                    "📝 Final Answer", "🔍 Intent & Metadata", "📄 Top Results"
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