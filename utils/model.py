import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

def build_bilstm(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dropout: float = 0.3,
):
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, mask_zero=True),
            Bidirectional(LSTM(lstm_units, return_sequences=False)),
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
