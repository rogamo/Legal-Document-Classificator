import streamlit as st

st.title("🧠 Inference Interface")

st.markdown("Enter your text below and get predictions from the model.")

# Placeholder for your inference logic
user_input = st.text_area("Input Text", placeholder="Type something here...")

if st.button("Predict"):
    # Replace with your actual model prediction code
    pred_class = "Happy 😊"
    confidence = 0.93

    st.success(f"Predicted Class: **{pred_class}**")
    st.info(f"Confidence: {confidence*100:.2f}%")
