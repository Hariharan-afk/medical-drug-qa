# Entrypoint for Hugging Face Spaces (Streamlit SDK expects app.py at repo root)
# This imports and executes the Streamlit app defined in app/main.py

from app.main import *  # noqa: F401,F403

