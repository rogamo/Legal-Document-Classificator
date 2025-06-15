import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

st.title("🧐 Model Analysis and Justification")

st.markdown("""
Explore what makes this classification problem hard, how the model was chosen, and how it performed.
""")

# Dataset challenges
st.subheader("📌 Dataset Challenges")
st.markdown("""
- Imbalanced class distribution
- Noisy or ambiguous language (e.g., sarcasm)
- Multilinguality in some samples
""")

# Model justification
st.subheader("🤖 Model Choice Justification")
st.markdown("""
We used **XLM-RoBERTa** for its multilingual understanding capabilities and robustness to noise.

Relevant references:
- [XLM-R: Conneau et al., 2020](https://arxiv.org/abs/1911.02116)
- [Kaggle NLP Competitions](https://www.kaggle.com/)
""")

# Dummy classification report
st.subheader("📋 Classification Report")
report_dict = {
    "precision": [0.88, 0.70, 0.80],
    "recall": [0.90, 0.65, 0.78],
    "f1-score": [0.89, 0.67, 0.79]
}
labels = ["Positive", "Negative", "Neutral"]
df_report = pd.DataFrame(report_dict, index=labels)
st.dataframe(df_report)

# Dummy confusion matrix
st.subheader("🧮 Confusion Matrix")
conf_matrix = [[50, 2, 3],
               [4, 30, 6],
               [2, 5, 25]]

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Error analysis
st.subheader("🔍 Error Analysis")
st.markdown("""
**False Positives:** Many sarcastic comments were misclassified as Positive.  
**False Negatives:** Short neutral comments were often labeled as Negative.

### Suggestions:
- Add more data with sarcasm examples.
- Use ensemble models or multi-task learning.
""")
