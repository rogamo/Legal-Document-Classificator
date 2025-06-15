import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.title("📊 Dataset Visualization")

st.markdown("Visualize important properties of the dataset, like class distribution and text length.")

# Dummy dataset (replace with your own)
# For now, simulate data
df = pd.DataFrame({
    'text': ["I love this!", "Terrible experience...", "Meh", "So happy", "Awful", "Nice", "Worst ever", "Good", "Horrible", "Perfect"],
    'label': ["Positive", "Negative", "Neutral", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", "Positive"]
})
df['length'] = df['text'].apply(len)

# Class distribution
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='label', ax=ax)
st.pyplot(fig)

# Token length distribution
st.subheader("Text Length Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['length'], bins=10, kde=True, ax=ax2)
st.pyplot(fig2)

# Word cloud
st.subheader("Word Cloud")
text_blob = ' '.join(df['text'])
wordcloud = WordCloud(background_color='white').generate(text_blob)
fig3, ax3 = plt.subplots()
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis("off")
st.pyplot(fig3)
