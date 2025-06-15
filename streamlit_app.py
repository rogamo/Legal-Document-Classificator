import streamlit as st

st.set_page_config(page_title="Text Classifier Dashboard", layout="centered")

st.title("🚀 Text Classification App")
st.subheader("Transformers + Streamlit for NLP Projects")

st.markdown("""
Welcome to the interactive dashboard for our text classification project!  
This app lets you explore and evaluate a fine-tuned transformer model built for NLP tasks.

### 🧭 App Sections
- **Inference Interface**: Try out the model with your own text.
- **Dataset Visualization**: Explore visual summaries and data insights.
- **Hyperparameter Tuning**: See how the model was optimized.
- **Model Analysis & Justification**: Understand model performance, limitations, and possible improvements.

---

**To get started**, select a page from the sidebar 👈
""")

st.info("Tip: If running locally, use `streamlit run streamlit_app.py`")

st.markdown("---")
st.caption("Built with ❤️ using HuggingFace Transformers and Streamlit.")
