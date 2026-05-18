
import pandas as pd
import numpy as np


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua indikator teknikal"""
    df = df.copy()

    # ── Moving Averages ──────────────────────────────────────
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── RSI ──────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Mid"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    )

    # ── Stochastic Oscillator ─────────────────────────────────
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # ── ATR ───────────────────────────────────────────────────
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["ATR"] = df["TR"].rolling(window=14).mean()

    # ── OBV ───────────────────────────────────────────────────
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # ── Volume MA ─────────────────────────────────────────────
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    # ── Williams %R ───────────────────────────────────────────
    df["Williams_R"] = -100 * (high_14 - df["Close"]) / (high_14 - low_14)

    # ── CCI ───────────────────────────────────────────────────
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_deviation = typical_price.rolling(window=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    df["CCI"] = (typical_price - typical_price.rolling(window=20).mean()) / (
        0.015 * mean_deviation
    )

    # ── ROC ───────────────────────────────────────────────────
    df["ROC"] = df["Close"].pct_change(periods=12) * 100

    return df


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """Ringkasan sinyal dari semua indikator"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {}

    # RSI Signal
    rsi = latest["RSI"]
    if rsi > 70:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "OVERBOUGHT ⚠️", "bias": "BEARISH"}
    elif rsi < 30:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "OVERSOLD 🟢", "bias": "BULLISH"}
    else:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "NORMAL", "bias": "NEUTRAL"}

    # MACD Signal
    macd_cross = "BULLISH" if latest["MACD"] > latest["MACD_Signal"] else "BEARISH"
    golden_cross = latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]
    dead_cross = latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]

    macd_signal_text = "GOLDEN CROSS 🚀" if golden_cross else ("DEAD CROSS ⚠️" if dead_cross else macd_cross)
    signals["MACD"] = {
        "value": round(latest["MACD"], 4),
        "signal": macd_signal_text,
        "bias": macd_cross,
    }

    # Bollinger Bands Signal
    bb_pos = latest["BB_Position"]
    if bb_pos > 0.9:
        bb_signal = "UPPER BAND (Overbought) ⚠️"
        bb_bias = "BEARISH"
    elif bb_pos < 0.1:
        bb_signal = "LOWER BAND (Oversold) 🟢"
        bb_bias = "BULLISH"
    else:
        bb_signal = f"MID BAND ({round(bb_pos*100, 1)}%)"
        bb_bias = "NEUTRAL"
    signals["Bollinger Bands"] = {
        "value": round(bb_pos, 3),
        "signal": bb_signal,
        "bias": bb_bias,
    }

    # Moving Average Signal
    price = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]

    if price > sma20 and price > sma50:
        ma_signal = "ABOVE MA20 & MA50 🚀"
        ma_bias = "BULLISH"
    elif price < sma20 and price < sma50:
        ma_signal = "BELOW MA20 & MA50 ⚠️"
        ma_bias = "BEARISH"
    else:
        ma_signal = "MIXED"
        ma_bias = "NEUTRAL"
    signals["Moving Average"] = {
        "value": round(price, 2),
        "signal": ma_signal,
        "bias": ma_bias,
    }

    # Stochastic Signal
    stoch_k = latest["Stoch_K"]
    if stoch_k > 80:
        stoch_signal = "OVERBOUGHT ⚠️"
        stoch_bias = "BEARISH"
    elif stoch_k < 20:
        stoch_signal = "OVERSOLD 🟢"
        stoch_bias = "BULLISH"
    else:
        stoch_signal = "NORMAL"
        stoch_bias = "NEUTRAL"
    signals["Stochastic"] = {
        "value": round(stoch_k, 2),
        "signal": stoch_signal,
        "bias": stoch_bias,
    }

    # Volume Signal
    vol_ratio = latest["Volume_Ratio"]
    if vol_ratio > 1.5:
        vol_signal = f"HIGH VOLUME ({round(vol_ratio, 2)}x) 🔥"
        vol_bias = "STRONG"
    elif vol_ratio < 0.5:
        vol_signal = f"LOW VOLUME ({round(vol_ratio, 2)}x)"
        vol_bias = "WEAK"
    else:
        vol_signal = f"NORMAL ({round(vol_ratio, 2)}x)"
        vol_bias = "NEUTRAL"
    signals["Volume"] = {
        "value": round(vol_ratio, 2),
        "signal": vol_signal,
        "bias": vol_bias,
    }

    # Overall bias count
    bullish_count = sum(1 for s in signals.values() if s["bias"] == "BULLISH")
    bearish_count = sum(1 for s in signals.values() if s["bias"] == "BEARISH")

    signals["_summary"] = {
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "overall": "BULLISH" if bullish_count > bearish_count else (
            "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
        ),
    }

    return signals
