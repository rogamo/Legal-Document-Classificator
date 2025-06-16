import streamlit as st
import optuna, json, os, pandas as pd, tensorflow as tf
from utils.data import load_fake_news
from utils.preprocessing import prepare_text, MAX_VOCAB
from utils.model import build_bilstm

st.header("🔧 Hyperparameter Tuning (Optuna)")

MODEL_PATH   = "models/bi_lstm_fake_news.h5"
PARAMS_PATH  = "models/bi_lstm_fake_news_params.json"

# ── CASE 1: model already trained ──────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    st.success("Model already trained ✅")

    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            best_params = json.load(f)
        st.subheader("🔝 Stored hyper-parameters")
        st.dataframe(
            pd.DataFrame(best_params, index=["value"]).T.reset_index() \
              .rename(columns={"index": "parameter"}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Params file not found — retrain below if you want them.")

    st.stop()          # ❗ early exit so rest of page doesn’t rerun

# ── CASE 2: need to run Optuna search ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_optuna(n_trials=10):
    df = load_fake_news()
    tok, X_tr, X_te, y_tr, y_te = prepare_text(df)

    def objective(trial):
        emb_dim    = trial.suggest_categorical("emb_dim", [64, 128, 256])
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        drop       = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        model = build_bilstm(MAX_VOCAB, emb_dim, lstm_units, drop)
        model.optimizer.learning_rate = lr

        hist = model.fit(
            X_tr, y_tr, epochs=3, batch_size=128,
            validation_split=0.2, verbose=0
        )
        return max(hist.history["val_accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

study = run_optuna()

# show best params
best_params = study.best_trial.params
st.subheader("🔝 Best Hyper-Parameters (new run)")
st.dataframe(
    pd.DataFrame(best_params, index=["value"]).T.reset_index() \
      .rename(columns={"index": "parameter"}),
    hide_index=True,
    use_container_width=True,
)

# retrain full model with best params
df = load_fake_news()
tok, X_tr, X_te, y_tr, y_te = prepare_text(df)
model = build_bilstm(
    MAX_VOCAB,
    best_params["emb_dim"],
    best_params["lstm_units"],
    best_params["dropout"],
)
model.optimizer.learning_rate = best_params["lr"]
model.fit(X_tr, y_tr, epochs=5, batch_size=128, validation_split=0.1, verbose=0)

# save model + params
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
with open(PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=2)

st.success(f"Model & params saved to **models/**. Reload the page to see them!")
