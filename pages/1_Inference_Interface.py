import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import MAX_LEN

st.header("📝 Fake-News Detector – Inference")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/bi_lstm_fake_news.h5")

@st.cache_resource
def load_tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

text = st.text_area("Paste a news headline or sentence:", height=100)
if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded)[0])
    label = "FAKE 🛑" if prob >= 0.5 else "REAL ✅"
    confidence = prob if label.startswith("FAKE") else 1 - prob
    st.metric("Prediction", label, delta=f"{confidence*100:.1f}% confidence")
