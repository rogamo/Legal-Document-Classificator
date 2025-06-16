import pandas as pd
from datasets import load_dataset

def load_fake_news(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("liar", split=split)
    df = ds.to_pandas()[["statement", "label"]].rename(columns={"statement": "text", "label": "target"})
    df["target"] = df["target"].apply(lambda x: 0 if x in [0, 1, 2] else 1)
    return df
