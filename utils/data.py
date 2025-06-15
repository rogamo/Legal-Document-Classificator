import pandas as pd
from datasets import load_dataset

def load_fake_news(split: str = "train") -> pd.DataFrame:
    """
    Loads the `liar` fake-news dataset from Hugging Face and returns a clean DataFrame.

    Labels: 0 = REAL, 1 = FAKE
    """
    ds = load_dataset("liar", split=split)      # small, so it fits Codespaces RAM
    df = ds.to_pandas()[["statement", "label"]].rename(
        columns={"statement": "text", "label": "target"}
    )
    df["target"] = df["target"].apply(lambda x: 0 if x in [0, 1, 2] else 1)  # map half-truth+real = 0
    return df
