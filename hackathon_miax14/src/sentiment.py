"""
Simple keyword-based news sentiment for MIAX14 Hackathon.

No external NLP libraries required — uses a curated finance lexicon.
Output: daily sentiment score per date, ready to merge into features.
"""

import re
import pandas as pd
import numpy as np

POSITIVE_WORDS = {
    "growth", "profit", "record", "beat", "strong", "surge", "rally",
    "gain", "rise", "jump", "boost", "upgrade", "bullish", "outperform",
    "expand", "innovation", "breakthrough", "launch", "new", "award",
    "dividend", "buy", "recommend", "positive", "success", "increase",
    "partnership", "acquisition", "deal", "patent",
}

NEGATIVE_WORDS = {
    "loss", "decline", "fall", "drop", "miss", "weak", "cut", "downgrade",
    "bearish", "underperform", "risk", "concern", "warning", "fraud",
    "lawsuit", "layoff", "bankrupt", "default", "crisis", "sell", "short",
    "investigate", "probe", "fine", "penalty", "recall", "delay", "fail",
    "debt", "deficit", "negative", "decrease", "shutdown",
}


def _score_headline(headline: str) -> float:
    if not isinstance(headline, str):
        return 0.0
    tokens = set(re.findall(r"[a-z]+", headline.lower()))
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def build_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by Date with:
    - news_count: total headlines that day
    - news_sentiment: mean sentiment score (-1 to +1)
    - news_sentiment_std: std of sentiment (uncertainty proxy)
    """
    df = news_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["score"] = df["Headline"].apply(_score_headline)

    agg = df.groupby("Date").agg(
        news_count=("score", "count"),
        news_sentiment=("score", "mean"),
        news_sentiment_std=("score", "std"),
    )
    agg["news_sentiment_std"] = agg["news_sentiment_std"].fillna(0)
    return agg
