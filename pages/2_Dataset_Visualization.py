import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils.data import load_fake_news
from utils.preprocessing import prepare_text

st.header("📊 Dataset Visualisation")

df = load_fake_news()

st.subheader("Class distribution")
fig, ax = plt.subplots()
sns.countplot(df, x="target", ax=ax)
ax.set_xticklabels(["REAL", "FAKE"])
st.pyplot(fig)

st.subheader("Token length")
tok, X_tr, *_ = prepare_text(df)
lengths = [len(x.split()) for x in df["text"]]
fig2, ax2 = plt.subplots()
sns.histplot(lengths, bins=30, ax=ax2)
st.pyplot(fig2)

st.subheader("Word cloud – FAKE news")
fake_text = " ".join(df[df["target"] == 1]["text"].tolist())
wc = WordCloud(width=800, height=400).generate(fake_text)
st.image(wc.to_array())
