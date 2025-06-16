import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB = 25000
MAX_LEN = 120

def prepare_text(df):
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
    tok.fit_on_texts(df["text"].values)
    X = tok.texts_to_sequences(df["text"].values)
    X = pad_sequences(X, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return tok, X_train, X_test, y_train, y_test
