import streamlit as st
import importlib

# ── Global page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake-News Classifier Dashboard",
    page_icon="📰",
    layout="centered",
)

# ── Landing content (appears only on the main URL) ──────────────────────────
st.title("📰 Fake-News Classification App")
st.markdown(
    """
This dashboard lets you explore a **Bidirectional LSTM** model trained on the
[LIAR](https://huggingface.co/datasets/liar) dataset to detect fake news.

**Navigate with the sidebar** to:
1. **Inference Interface** – paste text and get predictions.  
2. **Dataset Visualization** – see class balance, token lengths, word cloud.  
3. **Hyperparameter Tuning** – run an Optuna search and train the best model.  
4. **Model Analysis** – precision / recall, confusion matrix, and error cases.
"""
)

st.info("Choose a page on the left to begin 👉")
