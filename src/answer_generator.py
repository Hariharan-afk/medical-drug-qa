# # src/answer_generator.py

# import requests
# from dotenv import load_dotenv
# import os

# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL = "llama3-8b-8192"  # or use "mixtral-8x7b-32768"
# load_dotenv() 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment

# def generate_answer(query: str, context_chunks: list, model: str = GROQ_MODEL) -> str:
#     """
#     Use Groq API to generate an answer using the query and retrieved chunks.
#     """
#     if not GROQ_API_KEY:
#         raise ValueError("GROQ_API_KEY not set in environment.")

#     context = "\n\n".join(context_chunks)
#     prompt = (
#     "You are a reliable medical assistant. "
#     "Use ONLY the information provided in the context below to answer the user's question. "
#     "If the answer is not explicitly available in the context, say you do not have enough information to answer.\n\n"
#     f"Context:\n{context}\n\n"
#     f"Question:\n{query}\n\n"
#     f"Answer:"
# )


#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": "You are a medical assistant who answers based on trusted drug documentation."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3,
#         "max_tokens": 2048
#     }

#     response = requests.post(GROQ_API_URL, headers=headers, json=payload)
#     response.raise_for_status()

#     return response.json()["choices"][0]["message"]["content"].strip()



# src/answer_generator.py

import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env
load_dotenv()

# Configuration: API endpoint and default model
GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # must be set in your environment


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    model: str = GROQ_MODEL
) -> str:
    """
    Generate a medical answer using the Groq API, based on provided context_chunks.

    Args:
        query: The user's question.
        context_chunks: A list of dicts, each containing at least a 'chunk_text' key.
        model: The model name to invoke (defaults to GROQ_MODEL).

    Returns:
        A string containing the assistant's answer.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        RuntimeError: If the API returns no valid response.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment.")

    # Extract the raw text from each chunk; support dicts or plain strings
    texts: List[str] = []
    for chunk in context_chunks:
        if isinstance(chunk, dict):
            txt = chunk.get("chunk_text", "").strip()
        else:
            txt = str(chunk).strip()
        if txt:
            texts.append(txt)

    # If no context is provided, we cannot answer
    if not texts:
        return "Sorry, I do not have enough information to answer that question."

    # Build a numbered context section for clarity
    context = "\n\n".join(f"Chunk {i+1}:\n{text}" for i, text in enumerate(texts))

    # System and user prompts
    system_msg = (
        "You are a medical assistant who answers based on trusted drug documentation."
    )
    user_msg = (
        "Use ONLY the information provided in the context below to answer the user's question. "
        "If the answer is not explicitly available in the context, say you do not have enough information to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Send the request
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract the assistant's content
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned from Groq API.")

    message = choices[0].get("message", {})
    answer = message.get("content", "").strip()
    return answer
