import pandas as pd
import numpy as np


def detect_anomaly(series, threshold=2):
    """
    Detect anomalies using Z-score method.

    Args:
        series: Pandas Series
        threshold: Z-score threshold

    Returns:
        dict containing anomaly information
    """

    # Remove null values
    series = series.dropna()

    if len(series) < 2:
        return {
            "is_anomaly": False,
            "reason": "Not enough data"
        }

    mean = series.mean()
    std = series.std()

    latest_value = series.iloc[-1]

    # Avoid division by zero
    if std == 0:
        z_score = 0
    else:
        z_score = (latest_value - mean) / std

    is_anomaly = abs(z_score) > threshold

    return {
        "latest_value": round(latest_value, 2),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "z_score": round(z_score, 2),
        "is_anomaly": is_anomaly
    }


def detect_sentiment_spike(sentiment_series):
    """
    Detect sudden negative sentiment spikes
    """

    result = detect_anomaly(sentiment_series)

    if result["is_anomaly"]:
        return (
            f"⚠️ Sentiment anomaly detected. "
            f"Latest value = {result['latest_value']}, "
            f"Z-score = {result['z_score']}"
        )

    return "✅ No major sentiment anomaly detected."