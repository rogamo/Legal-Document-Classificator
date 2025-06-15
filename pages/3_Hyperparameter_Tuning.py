import streamlit as st
import optuna
import tensorflow as tf
from utils.data import load_fake_news
from utils.preprocessing import prepare_text, MAX_VOCAB
from utils.model import build_bilstm
import os

st.header("🔧 Hyperparameter Tuning (Optuna)")

MODEL_PATH = "models/bi_lstm_fake_news.h5"
if os.path.exists(MODEL_PATH):
    st.success("Model already trained. ✅")
    st.stop()

@st.cache_data(show_spinner=False)
def run_optuna(n_trials=10):
    df = load_fake_news()
    tok, X_tr, X_te, y_tr, y_te = prepare_text(df)

    def objective(trial):
        emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        drop = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        model = build_bilstm(MAX_VOCAB, emb_dim, lstm_units, drop)
        model.optimizer.learning_rate = lr

        hist = model.fit(
            X_tr, y_tr,
            epochs=3, batch_size=128,
            validation_split=0.2,
            verbose=0
        )
        return max(hist.history["val_accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

study = run_optuna()

st.subheader("Best Trial")
st.json(study.best_trial.params)

# Train best model
df = load_fake_news()
tok, X_tr, X_te, y_tr, y_te = prepare_text(df)
best = study.best_trial.params
model = build_bilstm(MAX_VOCAB, best["emb_dim"], best["lstm_units"], best["dropout"])
model.optimizer.learning_rate = best["lr"]
model.fit(X_tr, y_tr, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
model.save(MODEL_PATH)
st.success(f"Model saved to {MODEL_PATH}")
