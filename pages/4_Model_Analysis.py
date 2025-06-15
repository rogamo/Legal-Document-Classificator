import streamlit as st
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils.preprocessing import prepare_text, MAX_LEN
from utils.data import load_fake_news
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.header("🧪 Evaluation & Error Analysis")

if not os.path.exists("models/bi_lstm_fake_news.h5"):
    st.warning("Train model first using Page 3.")
    st.stop()

model = tf.keras.models.load_model("models/bi_lstm_fake_news.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

df = load_fake_news()
tok, X_tr, X_te, y_tr, y_te = prepare_text(df)
y_pred_prob = model.predict(X_te).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

st.subheader("Classification Report")
report = classification_report(y_te, y_pred, target_names=["REAL", "FAKE"], output_dict=True)
st.dataframe(report)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Examples of Errors")
for idx in np.where(y_te != y_pred)[0][:5]:
    st.markdown(f"- **Text**: {df.iloc[idx]['text']}")
    st.markdown(f"  - True: {['REAL','FAKE'][y_te[idx]]}, Predicted: {['REAL','FAKE'][y_pred[idx]]}, Prob: {y_pred_prob[idx]:.2f}")
