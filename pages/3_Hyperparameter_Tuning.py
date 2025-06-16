# pages/3_Hyperparameter_Tuning.py
import os, json, optuna, pandas as pd, tensorflow as tf, streamlit as st
from optuna.visualization import plot_optimization_history

from utils.data          import load_fake_news
from utils.preprocessing import prepare_text, MAX_VOCAB
from utils.model         import build_bilstm

st.header("🔧 Hyperparameter Tuning (Optuna)")

MODEL_PATH  = "models/bi_lstm_fake_news.h5"
PARAMS_PATH = "models/bi_lstm_fake_news_params.json"
N_TRIALS    = 10                         # easy to change from the UI if you wish

# ────────────────────────────────
# 1) If we already trained before
# ────────────────────────────────
if os.path.exists(MODEL_PATH):
    st.success("Model already trained ✅")

    if os.path.exists(PARAMS_PATH):
        best_params = json.load(open(PARAMS_PATH))
        st.subheader("🔝 Stored hyper-parameters")
        st.dataframe(
            pd.DataFrame(best_params, index=["value"]).T.reset_index()
              .rename(columns={"index": "parameter"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Params file not found — retrain below if you want them.")

    st.stop()   # ← Nothing else to do on this run

# ────────────────────────────────
# 2) Search with Optuna otherwise
# ────────────────────────────────
@st.cache_data(show_spinner=False)
def run_optuna(n_trials=N_TRIALS):
    df = load_fake_news()
    tok, X_tr, X_te, y_tr, y_te = prepare_text(df)

    def objective(trial):
        emb_dim    = trial.suggest_categorical("emb_dim",    [64, 128, 256])
        lstm_units = trial.suggest_int       ("lstm_units",  32, 128, step=32)
        drop       = trial.suggest_float     ("dropout",     0.1, 0.5, step=0.1)
        lr         = trial.suggest_float     ("lr",          1e-4, 5e-3, log=True)

        model = build_bilstm(MAX_VOCAB, emb_dim, lstm_units, drop)
        model.optimizer.learning_rate = lr

        hist = model.fit(
            X_tr, y_tr,
            epochs=3, batch_size=128,
            validation_split=0.2,
            verbose=0,
        )
        return max(hist.history["val_accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

study       = run_optuna()
best_params = study.best_trial.params

# ────────────  UI OUTPUTS  ────────────
st.subheader("🔝 Best Hyper-Parameters (new run)")
st.dataframe(
    pd.DataFrame(best_params, index=["value"]).T.reset_index()
      .rename(columns={"index": "parameter"}),
    use_container_width=True, hide_index=True,
)

st.markdown("""
**Parameters tuned**

| Parameter        | Search Space                |
|------------------|-----------------------------|
| `emb_dim`        | {64, 128, 256}              |
| `lstm_units`     | {32, 64, 96, 128}           |
| `dropout`        | 0.1 – 0.5 (step 0.1)        |
| `lr`             | 1e-4 – 5e-3 (log-uniform)   |
| `_epochs_`       | fixed at **5**              |
| `_batch_size_`   | fixed at **128**            |
""")

st.subheader("📈 Optimization History")
st.plotly_chart(plot_optimization_history(study), use_container_width=True)

# ────────────────────────────────
# 3) Retrain full model w/ best hp
# ────────────────────────────────
df  = load_fake_news()
tok, X_tr, X_te, y_tr, y_te = prepare_text(df)

model = build_bilstm(
    MAX_VOCAB,
    best_params["emb_dim"],
    best_params["lstm_units"],
    best_params["dropout"],
)
model.optimizer.learning_rate = best_params["lr"]

model.fit(X_tr, y_tr, epochs=5, batch_size=128, validation_split=0.1, verbose=0)

# save artefacts
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
json.dump(best_params, open(PARAMS_PATH, "w"), indent=2)

st.success("🎉 Model & params saved → `models/`.  Reload this page to see the stored table.")
