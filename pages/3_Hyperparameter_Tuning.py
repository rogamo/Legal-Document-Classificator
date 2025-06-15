import streamlit as st
import matplotlib.pyplot as plt

st.title("⚙️ Hyperparameter Tuning")

st.markdown("""
This page shows how the best model configuration was selected using tools like **Optuna** or **Keras Tuner**.
""")

# Static section — replace with dynamic content later if needed
st.subheader("🔧 Parameters Tuned")
st.markdown("""
- Learning rate
- Batch size
- Dropout rate
- Number of epochs
""")

# Best config (dummy values)
st.subheader("🏆 Best Configuration")
best_config = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "dropout_rate": 0.3,
    "epochs": 4
}
st.json(best_config)

# Dummy performance graph
st.subheader("📈 Trial Performance Over Time")
fig, ax = plt.subplots()
ax.plot([0.7, 0.73, 0.75, 0.76, 0.78], label="F1 Score")
ax.set_xlabel("Trial")
ax.set_ylabel("F1 Score")
ax.set_title("Optimization Trials")
ax.legend()
st.pyplot(fig)
