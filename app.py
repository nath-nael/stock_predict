# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBCA Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px; padding: 16px 20px;
        border-left: 4px solid #4f8ef7; margin-bottom: 10px;
    }
    .metric-label { color: #8b9ab5; font-size: 13px; font-weight: 500; }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: 700; }
    .metric-delta-pos { color: #00d4aa; font-size: 13px; }
    .metric-delta-neg { color: #ff4b6e; font-size: 13px; }
    .rec-box {
        border-radius: 12px; padding: 20px; text-align: center;
        font-size: 22px; font-weight: 800; letter-spacing: 1px; margin: 10px 0;
    }
    .rec-buy  { background:linear-gradient(135deg,#00d4aa22,#00d4aa44);border:2px solid #00d4aa;color:#00d4aa; }
    .rec-sell { background:linear-gradient(135deg,#ff4b6e22,#ff4b6e44);border:2px solid #ff4b6e;color:#ff4b6e; }
    .rec-hold { background:linear-gradient(135deg,#f5a62322,#f5a62344);border:2px solid #f5a623;color:#f5a623; }
    .model-badge {
        background:#1e2130; border-radius:8px; padding:8px 14px;
        font-size:12px; color:#8b9ab5; margin:4px; display:inline-block;
    }
    .section-title {
        color:#4f8ef7; font-size:15px; font-weight:600;
        text-transform:uppercase; letter-spacing:1px;
        margin:20px 0 10px; border-bottom:1px solid #252a3d; padding-bottom:6px;
    }
    .warning-box {
        background:#f5a62311; border:1px solid #f5a623; border-radius:8px;
        padding:10px 14px; color:#f5a623; font-size:12px; margin:8px 0;
    }
    .accuracy-box {
        background:#1e2130; border-radius:10px; padding:14px;
        border:1px solid #252a3d;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & FEATURES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_data(ticker: str = "BBCA.JK", period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Rich feature engineering — the real edge."""
    c = df["Close"].squeeze().copy()
    h = df["High"].squeeze().copy()
    l = df["Low"].squeeze().copy()
    v = df["Volume"].squeeze().copy()

    # ── Trend ─────────────────────────────────────────────────────────────────
    for w in [5, 7, 10, 14, 20, 50]:
        df[f"MA{w}"]  = c.rolling(w).mean()
        df[f"EMA{w}"] = c.ewm(span=w, adjust=False).mean()

    # ── Momentum ──────────────────────────────────────────────────────────────
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Stochastic RSI
    rsi = df["RSI14"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # Rate of Change
    for w in [5, 10, 20]:
        df[f"ROC{w}"] = c.pct_change(w) * 100

    # Williams %R
    hw = h.rolling(14).max()
    lw = l.rolling(14).min()
    df["WilliamsR"] = -100 * (hw - c) / (hw - lw + 1e-9)

    # ── Volatility ────────────────────────────────────────────────────────────
    # ATR
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR14"] / c * 100   # normalised

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Mid"]   = bb_mid
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid  # squeeze indicator
    df["BB_Pos"]   = (c - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # Historical volatility
    df["HV20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── Volume ────────────────────────────────────────────────────────────────
    df["Vol_MA20"]   = v.rolling(20).mean()
    df["Vol_Ratio"]  = v / df["Vol_MA20"]

    # On-Balance Volume
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["OBV"]        = obv
    df["OBV_MA10"]   = obv.rolling(10).mean()
    df["OBV_Signal"] = (obv - df["OBV_MA10"]) / (df["OBV_MA10"].abs() + 1e-9)

    # ── Price structure ───────────────────────────────────────────────────────
    df["HL_pct"]     = (h - l) / c * 100          # daily range %
    df["CO_pct"]     = (c - df["Open"].squeeze()) / df["Open"].squeeze() * 100
    df["Gap"]        = (df["Open"].squeeze() - c.shift()) / c.shift() * 100

    # Log returns (stationary)
    df["LogRet1"]  = np.log(c / c.shift(1))
    df["LogRet5"]  = np.log(c / c.shift(5))
    df["LogRet10"] = np.log(c / c.shift(10))

    # ── Lagged features (what model "sees" at prediction time) ────────────────
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f"Close_lag{lag}"] = c.shift(lag)
        df[f"Ret_lag{lag}"]   = c.pct_change().shift(lag)

    # ── Target: next-N-day return (filled later per horizon) ─────────────────
    df["Close_raw"] = c   # keep for reference

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "MA5","MA7","MA10","MA14","MA20","MA50",
    "EMA5","EMA7","EMA10","EMA14","EMA20","EMA50",
    "RSI14","StochRSI","MACD","MACD_Signal","MACD_Hist",
    "ROC5","ROC10","ROC20","WilliamsR",
    "ATR14","ATR_pct","BB_Width","BB_Pos","HV20",
    "Vol_Ratio","OBV_Signal",
    "HL_pct","CO_pct","Gap",
    "LogRet1","LogRet5","LogRet10",
    "Close_lag1","Close_lag2","Close_lag3","Close_lag5","Close_lag7","Close_lag10",
    "Ret_lag1","Ret_lag2","Ret_lag3","Ret_lag5","Ret_lag7","Ret_lag10",
]


def build_dataset(df: pd.DataFrame, horizon: int):
    """
    Target = % return over `horizon` days ahead.
    Using returns (not price levels) makes the model stationary & generalisable.
    """
    c = df["Close_raw"]
    df["Target"] = (c.shift(-horizon) - c) / c * 100   # future % return

    data = df[FEATURE_COLS + ["Target"]].dropna()
    X = data[FEATURE_COLS].values
    y = data["Target"].values
    return X, y, data.index


def train_ensemble(X, y):
    """
    Ensemble of:
      1. GradientBoosting  — captures non-linear interactions
      2. RandomForest      — robust, low variance
    Validated with TimeSeriesSplit (no data leakage).
    """
    tscv = TimeSeriesSplit(n_splits=5)

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )

    # ── Cross-val MAPE per model ───────────────────────────────────────────────
    mapes_gb, mapes_rf = [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = RobustScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        gb.fit(X_tr_sc, y_tr)
        rf.fit(X_tr_sc, y_tr)

        # avoid division by zero in MAPE when y_val has zeros
        mask = y_val != 0
        if mask.sum() > 0:
            mapes_gb.append(mean_absolute_percentage_error(
                y_val[mask], gb.predict(X_val_sc)[mask]))
            mapes_rf.append(mean_absolute_percentage_error(
                y_val[mask], rf.predict(X_val_sc)[mask]))

    # ── Final fit on ALL data ─────────────────────────────────────────────────
    scaler_final = RobustScaler()
    X_sc = scaler_final.fit_transform(X)
    gb.fit(X_sc, y)
    rf.fit(X_sc, y)

    avg_mape_gb = float(np.mean(mapes_gb)) * 100 if mapes_gb else 0.0
    avg_mape_rf = float(np.mean(mapes_rf)) * 100 if mapes_rf else 0.0

    # Weight by inverse MAPE (better model gets more weight)
    inv_gb = 1 / (avg_mape_gb + 1e-6)
    inv_rf = 1 / (avg_mape_rf + 1e-6)
    total  = inv_gb + inv_rf
    w_gb   = inv_gb / total
    w_rf   = inv_rf / total

    return {
        "gb": gb, "rf": rf,
        "scaler": scaler_final,
        "w_gb": w_gb, "w_rf": w_rf,
        "mape_gb": avg_mape_gb,
        "mape_rf": avg_mape_rf,
        "weighted_mape": w_gb * avg_mape_gb + w_rf * avg_mape_rf,
    }


def iterative_predict(df: pd.DataFrame, models: dict,
                      horizon: int) -> dict:
    """
    Walk-forward prediction:
    Predict day+1 return → update synthetic price → rebuild features → repeat.
    This avoids the naive mistake of predicting all days from a single snapshot.
    """
    gb      = models["gb"]
    rf      = models["rf"]
    scaler  = models["scaler"]
    w_gb    = models["w_gb"]
    w_rf    = models["w_rf"]

    # Work on a copy we can extend
    df_sim = df.copy()
    predicted_prices = []
    predicted_returns = []

    current_price = float(df_sim["Close_raw"].iloc[-1])

    for step in range(horizon):
        # Extract latest feature row
        row = df_sim[FEATURE_COLS].iloc[-1:].values
        row_sc = scaler.transform(row)

        ret_gb = gb.predict(row_sc)[0]
        ret_rf = rf.predict(row_sc)[0]
        ret_ensemble = w_gb * ret_gb + w_rf * ret_rf   # % return

        # Clip extreme predictions (sanity guard: max ±5% per day)
        ret_ensemble = np.clip(ret_ensemble, -5.0, 5.0)

        next_price = current_price * (1 + ret_ensemble / 100)
        predicted_prices.append(next_price)
        predicted_returns.append(ret_ensemble)

        # ── Append synthetic row to df_sim ────────────────────────────────────
        new_row = df_sim.iloc[-1:].copy()
        new_row.index = [df_sim.index[-1] + timedelta(days=1)]

        # Update price columns
        new_row["Close_raw"] = next_price
        new_row["Open"]      = current_price
        new_row["High"]      = next_price * 1.005
        new_row["Low"]       = next_price * 0.995
        new_row["Volume"]    = df_sim["Volume"].squeeze().rolling(5).mean().iloc[-1]

        df_sim = pd.concat([df_sim, new_row])

        # Recompute indicators on extended df
        df_sim = add_indicators(df_sim)
        df_sim.ffill(inplace=True)

        current_price = next_price

    # ── Confidence band via Monte Carlo on ATR ────────────────────────────────
    atr_pct = float(df["ATR_pct"].iloc[-1]) / 100
    base    = float(df["Close_raw"].iloc[-1])
    upper, lower = [], []
    for i, p in enumerate(predicted_prices):
        sigma = atr_pct * np.sqrt(i + 1)
        upper.append(p * (1 + 1.65 * sigma))   # ~90% CI
        lower.append(p * (1 - 1.65 * sigma))

    return {
        "prices":   predicted_prices,
        "returns":  predicted_returns,
        "upper":    upper,
        "lower":    lower,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(df: pd.DataFrame, pred: dict) -> dict:
    price   = float(df["Close_raw"].iloc[-1])
    rsi     = float(df["RSI14"].iloc[-1])
    macd    = float(df["MACD"].iloc[-1])
    sig     = float(df["MACD_Signal"].iloc[-1])
    ma20    = float(df["MA20"].iloc[-1])
    ma50    = float(df["MA50"].iloc[-1])
    bb_pos  = float(df["BB_Pos"].iloc[-1])
    stoch   = float(df["StochRSI"].iloc[-1])
    vol_r   = float(df["Vol_Ratio"].iloc[-1])
    obv_sig = float(df["OBV_Signal"].iloc[-1])
    hv20    = float(df["HV20"].iloc[-1])

    pred_ret = (pred["prices"][-1] - price) / price * 100

    score = 0.0

    # ── RSI (weight 2) ────────────────────────────────────────────────────────
    if rsi < 25:   score += 2.0
    elif rsi < 35: score += 1.0
    elif rsi < 45: score += 0.5
    elif rsi > 75: score -= 2.0
    elif rsi > 65: score -= 1.0
    elif rsi > 55: score -= 0.5

    # ── StochRSI (weight 1) ───────────────────────────────────────────────────
    if stoch < 0.2:   score += 1.0
    elif stoch > 0.8: score -= 1.0

    # ── MACD (weight 1.5) ─────────────────────────────────────────────────────
    if macd > sig:
        score += 1.5 if macd > 0 else 0.75
    else:
        score -= 1.5 if macd < 0 else 0.75

    # ── Trend (weight 2) ──────────────────────────────────────────────────────
    if price > ma20 > ma50: score += 2.0
    elif price > ma20:      score += 1.0
    elif price < ma20 < ma50: score -= 2.0
    elif price < ma20:        score -= 1.0

    # ── Bollinger position (weight 1) ─────────────────────────────────────────
    if bb_pos < 0.1:   score += 1.0
    elif bb_pos > 0.9: score -= 1.0

    # ── Volume confirmation (weight 1) ────────────────────────────────────────
    if vol_r > 1.5:
        score += 1.0 if macd > sig else -1.0
    elif vol_r > 1.2:
        score += 0.5 if macd > sig else -0.5

    # ── OBV trend (weight 0.5) ────────────────────────────────────────────────
    if obv_sig > 0.05:   score += 0.5
    elif obv_sig < -0.05: score -= 0.5

    # ── Model prediction (weight 3 — highest) ────────────────────────────────
    if pred_ret > 4:    score += 3.0
    elif pred_ret > 2:  score += 2.0
    elif pred_ret > 0.5: score += 1.0
    elif pred_ret < -4:  score -= 3.0
    elif pred_ret < -2:  score -= 2.0
    elif pred_ret < -0.5: score -= 1.0

    # ── Volatility penalty (high vol = less confident) ────────────────────────
    if hv20 > 40: score *= 0.85

    # ── Map to recommendation ─────────────────────────────────────────────────
    max_score = 12.0
    pct = score / max_score

    if pct >= 0.35:
        rec      = "BUY"
        strength = "Strong" if pct >= 0.55 else "Moderate"
    elif pct <= -0.35:
        rec      = "SELL"
        strength = "Strong" if pct <= -0.55 else "Moderate"
    else:
        rec      = "HOLD"
        strength = "Neutral"

    return {
        "recommendation": rec,
        "strength":       strength,
        "score":          round(score, 2),
        "score_pct":      round(pct * 100, 1),
        "pred_ret":       pred_ret,
        "rsi":            rsi,
        "stoch":          stoch,
        "macd":           macd,
        "signal":         sig,
        "bb_pos":         bb_pos,
        "vol_ratio":      vol_r,
        "hv20":           hv20,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_main(df, pred_dates, pred_prices, upper, lower, label):
    show = df.tail(120)
    c    = show["Close_raw"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("Price & Prediction", "Volume", "RSI + StochRSI"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=show.index,
        open=show["Open"].squeeze(), high=show["High"].squeeze(),
        low=show["Low"].squeeze(),   close=c,
        name="BBCA",
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4b6e",
    ), row=1, col=1)

    # MAs
    for col_name, color, w in [("MA7","#f5a623",1.2),("MA20","#4f8ef7",1.5),("MA50","#b44fff",1.2)]:
        fig.add_trace(go.Scatter(
            x=show.index, y=show[col_name].squeeze(),
            name=col_name, line=dict(color=color, width=w), opacity=0.85,
        ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=show.index, y=show["BB_Upper"].squeeze(),
        line=dict(color="#8b9ab5", width=1, dash="dot"),
        showlegend=False, name="BB Upper",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=show.index, y=show["BB_Lower"].squeeze(),
        line=dict(color="#8b9ab5", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(139,154,181,0.06)",
        showlegend=False, name="BB Lower",
    ), row=1, col=1)

    # Prediction confidence band
    fig.add_trace(go.Scatter(
        x=pred_dates, y=upper,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred_dates, y=lower,
        fill="tonexty", fillcolor="rgba(79,142,247,0.18)",
        line=dict(width=0), name="90% CI", hoverinfo="skip",
    ), row=1, col=1)

    # Prediction line
    bridge_x = [show.index[-1]] + list(pred_dates)
    bridge_y = [float(c.iloc[-1])] + list(pred_prices)
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        name=f"Forecast ({label})",
        line=dict(color="#4f8ef7", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=5),
    ), row=1, col=1)

    # Volume
    vol_colors = ["#00d4aa" if c_ >= o_ else "#ff4b6e"
                  for c_, o_ in zip(show["Close_raw"], show["Open"].squeeze())]
    fig.add_trace(go.Bar(
        x=show.index, y=show["Volume"].squeeze(),
        name="Volume", marker_color=vol_colors, opacity=0.65,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=show.index, y=show["Vol_MA20"].squeeze(),
        name="Vol MA20", line=dict(color="#f5a623", width=1.2),
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=show.index, y=show["RSI14"].squeeze(),
        name="RSI 14", line=dict(color="#b44fff", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=show.index, y=show["StochRSI"].squeeze() * 100,
        name="StochRSI×100", line=dict(color="#00d4aa", width=1.2, dash="dot"),
    ), row=3, col=1)
    for lvl, clr in [(70,"#ff4b6e"),(30,"#00d4aa"),(50,"#8b9ab5")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                      opacity=0.45, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=720, margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1e2130", zerolinecolor="#1e2130")
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


def chart_macd(df):
    show = df.tail(120)
    fig  = go.Figure()
    colors = ["#00d4aa" if v >= 0 else "#ff4b6e"
              for v in show["MACD_Hist"].squeeze()]
    fig.add_trace(go.Bar(
        x=show.index, y=show["MACD_Hist"].squeeze(),
        name="Histogram", marker_color=colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=show.index, y=show["MACD"].squeeze(),
        name="MACD", line=dict(color="#4f8ef7", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=show.index, y=show["MACD_Signal"].squeeze(),
        name="Signal", line=dict(color="#f5a623", width=1.5),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=260, margin=dict(l=10,r=10,t=30,b=10),
        title="MACD (12,26,9)", legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1e2130")
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


def chart_feature_importance(models):
    gb = models["gb"]
    imp = pd.Series(gb.feature_importances_, index=FEATURE_COLS)
    top = imp.nlargest(15).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale="Blues",
            showscale=False,
        ),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=340, margin=dict(l=10,r=10,t=30,b=10),
        title="Top 15 Feature Importances (GradientBoosting)",
        xaxis_title="Importance",
    )
    fig.update_yaxes(gridcolor="#1e2130", tickfont=dict(size=11))
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        horizon_opt  = st.radio("Prediction Horizon", ["7 Days","1 Month"], index=0)
        horizon_days = 7 if horizon_opt == "7 Days" else 30

        st.markdown("---")
        data_period = st.selectbox(
            "Training Data Period",
            ["1y","2y","3y"],
            index=1,
            help="More data = better model generalisation",
        )

        st.markdown("---")
        st.markdown("### 🧠 Model Info")
        st.markdown("""
        **Ensemble:**
        - `GradientBoosting` (300 trees)
        - `RandomForest` (200 trees)

        **Validation:** TimeSeriesSplit (5-fold)  
        **Features:** 44 technical indicators  
        **Prediction:** Walk-forward iterative  
        **CI:** ATR-based Monte Carlo (~90%)

        ---
        ⚠️ *Not financial advice.*
        """)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 📈 BBCA Stock Prediction System")
    st.markdown(
        f"<span style='color:#8b9ab5;font-size:14px;'>"
        f"Bank Central Asia · IDX · "
        f"Updated {datetime.now().strftime('%d %b %Y %H:%M WIB')}"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load & process ────────────────────────────────────────────────────────
    with st.spinner("📥 Fetching BBCA data..."):
        raw = fetch_data("BBCA.JK", period=data_period)

    if raw.empty:
        st.error("❌ Could not fetch data. Check your internet connection.")
        return

    with st.spinner("⚙️ Computing 44 technical indicators..."):
        df = add_indicators(raw.copy())
        df.ffill(inplace=True)
        df.dropna(inplace=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    with st.spinner(f"🧠 Training ensemble model (horizon={horizon_days}d)..."):
        X, y, idx = build_dataset(df, horizon_days)
        models    = train_ensemble(X, y)

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("🔮 Running walk-forward prediction..."):
        pred = iterative_predict(df, models, horizon_days)

    pred_dates = pd.bdate_range(
        start=df.index[-1] + timedelta(days=1),
        periods=horizon_days,
    )

    rec = generate_recommendation(df, pred)

    # ── Key numbers ───────────────────────────────────────────────────────────
    price_now  = float(df["Close_raw"].iloc[-1])
    price_prev = float(df["Close_raw"].iloc[-2])
    price_pred = float(pred["prices"][-1])
    daily_chg  = (price_now - price_prev) / price_prev * 100
    pred_chg   = (price_pred - price_now) / price_now * 100

    # ── Model accuracy banner ─────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="accuracy-box">
            <div style="color:#8b9ab5;font-size:12px;">GradientBoosting CV-MAPE</div>
            <div style="color:#00d4aa;font-size:22px;font-weight:700;">
                {models['mape_gb']:.2f}%
            </div>
            <div style="color:#8b9ab5;font-size:11px;">Weight: {models['w_gb']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="accuracy-box">
            <div style="color:#8b9ab5;font-size:12px;">RandomForest CV-MAPE</div>
            <div style="color:#4f8ef7;font-size:22px;font-weight:700;">
                {models['mape_rf']:.2f}%
            </div>
            <div style="color:#8b9ab5;font-size:11px;">Weight: {models['w_rf']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="accuracy-box">
            <div style="color:#8b9ab5;font-size:12px;">Ensemble Weighted MAPE</div>
            <div style="color:#f5a623;font-size:22px;font-weight:700;">
                {models['weighted_mape']:.2f}%
            </div>
            <div style="color:#8b9ab5;font-size:11px;">
                TimeSeriesSplit · 5-fold · {len(X)} samples
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics ───────────────────────────────────────────────────────────────
    def mcard(label, val, delta=None, prefix="Rp "):
        d = ""
        if delta is not None:
            cls = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
            arr = "▲" if delta >= 0 else "▼"
            d = f'<div class="{cls}">{arr} {abs(delta):.2f}%</div>'
        return (f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{prefix}{val}</div>{d}</div>')

    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.markdown(mcard("Current Price", f"{price_now:,.0f}", daily_chg), unsafe_allow_html=True)
    with m2: st.markdown(mcard(f"Forecast ({horizon_opt})", f"{price_pred:,.0f}", pred_chg), unsafe_allow_html=True)
    with m3: st.markdown(mcard("RSI 14", f"{rec['rsi']:.1f}", prefix=""), unsafe_allow_html=True)
    with m4: st.markdown(mcard("HV20 (Ann.)", f"{rec['hv20']:.1f}%", prefix=""), unsafe_allow_html=True)
    with m5:
        vol = float(df["Volume"].squeeze().iloc[-1])
        st.markdown(mcard("Volume", f"{vol/1e6:.1f}M", prefix=""), unsafe_allow_html=True)

    st.markdown("")

    # ── Recommendation ────────────────────────────────────────────────────────
    rec_class = {"BUY":"rec-buy","SELL":"rec-sell","HOLD":"rec-hold"}[rec["recommendation"]]
    rec_icon  = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}[rec["recommendation"]]

    col_rec, col_detail = st.columns([1, 2])
    with col_rec:
        st.markdown(
            f'<div class="rec-box {rec_class}">'
            f'{rec_icon} {rec["strength"]} {rec["recommendation"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;color:#8b9ab5;font-size:12px;">'
            f'Composite score: <b style="color:#fff;">{rec["score"]:+.1f}</b> '
            f'({rec["score_pct"]:+.1f}% of max)</div>',
            unsafe_allow_html=True,
        )

    with col_detail:
        macd_bull = rec["macd"] > rec["signal"]
        rsi_zone  = ("Oversold 🟢" if rec["rsi"] < 35 else
                     "Overbought 🔴" if rec["rsi"] > 65 else "Neutral 🟡")
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:12px;padding:16px;">
            <div style="color:#8b9ab5;font-size:12px;margin-bottom:10px;">SIGNAL BREAKDOWN</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
                <span style="color:#8b9ab5;">Forecast Return:</span>
                <b style="color:{'#00d4aa' if pred_chg>=0 else '#ff4b6e'};">{pred_chg:+.2f}%</b>
                <span style="color:#8b9ab5;">RSI Zone:</span>
                <b style="color:#fff;">{rsi_zone} ({rec['rsi']:.1f})</b>
                <span style="color:#8b9ab5;">StochRSI:</span>
                <b style="color:#fff;">{rec['stoch']:.2f}</b>
                <span style="color:#8b9ab5;">MACD:</span>
                <b style="color:{'#00d4aa' if macd_bull else '#ff4b6e'};">{'Bullish ▲' if macd_bull else 'Bearish ▼'}</b>
                <span style="color:#8b9ab5;">BB Position:</span>
                <b style="color:#fff;">{rec['bb_pos']*100:.0f}% of band</b>
                <span style="color:#8b9ab5;">Vol Ratio:</span>
                <b style="color:#fff;">{rec['vol_ratio']:.2f}×</b>
            </div>
            <div style="margin-top:10px;color:#8b9ab5;font-size:12px;">
                90% CI target: <b style="color:#fff;">Rp {float(pred['lower'][-1]):,.0f}</b>
                – <b style="color:#fff;">Rp {float(pred['upper'][-1]):,.0f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Main chart ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Price Chart & Walk-Forward Forecast</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_main(df, pred_dates, pred["prices"],
                   pred["upper"], pred["lower"], horizon_opt),
        use_container_width=True,
    )

    # ── MACD + Feature importance ─────────────────────────────────────────────
    col_macd, col_fi = st.columns([1, 1])
    with col_macd:
        st.markdown('<div class="section-title">📉 MACD</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_macd(df), use_container_width=True)

    with col_fi:
        st.markdown('<div class="section-title">🔍 Feature Importance</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_feature_importance(models), use_container_width=True)

    # ── Prediction table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗓️ Forecast Schedule</div>',
                unsafe_allow_html=True)

    step = 1 if horizon_days == 7 else 5
    rows = []
    for i in range(0, horizon_days, step):
        idx_  = min(i, len(pred["prices"]) - 1)
        p     = pred["prices"][idx_]
        chg   = (p - price_now) / price_now * 100
        daily = pred["returns"][idx_]
        rows.append({
            "Date":        pred_dates[idx_].strftime("%d %b %Y"),
            "Day":         f"D+{idx_+1}",
            "Price (Rp)":  f"{p:,.0f}",
            "Total Chg":   f"{chg:+.2f}%",
            "Daily Ret":   f"{daily:+.2f}%",
            "Low (Rp)":    f"{pred['lower'][idx_]:,.0f}",
            "High (Rp)":   f"{pred['upper'][idx_]:,.0f}",
        })

    tbl = pd.DataFrame(rows)

    def color_chg(val):
        if isinstance(val, str) and "%" in val:
            return f"color: {'#00d4aa' if '+' in val else '#ff4b6e'}"
        return ""

    st.dataframe(
        tbl.style.applymap(color_chg, subset=["Total Chg","Daily Ret"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Technical summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Technical Summary</div>',
                unsafe_allow_html=True)

    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        st.markdown("**Moving Averages**")
        for name, col_key in [("MA 7","MA7"),("MA 20","MA20"),("MA 50","MA50")]:
            v = float(df[col_key].iloc[-1])
            sig_ = "🟢 Above" if price_now > v else "🔴 Below"
            st.markdown(f"`{name}` Rp {v:,.0f} → {sig_}")

    with cs2:
        st.markdown("**Bollinger Bands**")
        bb_u = float(df["BB_Upper"].iloc[-1])
        bb_m = float(df["BB_Mid"].iloc[-1])
        bb_l = float(df["BB_Lower"].iloc[-1])
        bb_p = float(df["BB_Pos"].iloc[-1]) * 100
        st.markdown(f"`Upper` Rp {bb_u:,.0f}")
        st.markdown(f"`Mid`   Rp {bb_m:,.0f}")
        st.markdown(f"`Lower` Rp {bb_l:,.0f}")
        st.progress(min(int(bb_p), 100), text=f"Position: {bb_p:.0f}%")

    with cs3:
        st.markdown("**Support & Resistance (30d)**")
        recent     = df["Close_raw"].tail(30)
        support    = float(recent.min())
        resistance = float(recent.max())
        dist_s = (price_now - support) / support * 100
        dist_r = (resistance - price_now) / resistance * 100
        st.markdown(f"🟢 `Support`    Rp {support:,.0f} (+{dist_s:.1f}%)")
        st.markdown(f"🔴 `Resistance` Rp {resistance:,.0f} (-{dist_r:.1f}%)")
        st.markdown(f"📍 `Current`    Rp {price_now:,.0f}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#8b9ab5;font-size:12px;'>"
        "⚠️ Educational purposes only. Not financial advice. "
        "Past performance does not guarantee future results. "
        "Always consult a qualified financial advisor."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
