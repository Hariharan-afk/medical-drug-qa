# chat_utils.py
from datetime import datetime
import streamlit as st

def append_to_history(query, intents, top_chunks, final_answer):
    """
    Appends a Q&A entry with timestamp to Streamlit session state history.
    """
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "query": query,
        "intents": intents,
        "top_chunks": [
            {"text": c.get('chunk_text'), "score": c.get('score')} 
            for c in top_chunks
        ],
        "answer": final_answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

def render_chat_history():
    """
    Renders the chat history with timestamps and user/bot messages.
    """
    if not st.session_state.history:
        return

    st.markdown("---")
    for entry in st.session_state.history:
        ts = entry.get("timestamp", "--")
        st.markdown(f"**{ts}** — **You:** {entry['query']}")
        st.markdown(f"**Bot:** {entry['answer']}")
        st.write("---")

def auto_scroll_to_bottom():
    """
    Scrolls to the bottom of the page after new messages.
    """
    st.markdown(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        unsafe_allow_html=True
    )
