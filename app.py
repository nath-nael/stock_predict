import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import ta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="📈 Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00C851, #007E33);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #00C851;
    }
    .rec-buy {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .rec-sell {
        background: linear-gradient(135deg, #FF4444, #CC0000);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .rec-hold {
        background: linear-gradient(135deg, #FFB300, #FF8800);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background: #262730;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)  # Cache 5 menit
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Ambil data saham dari Yahoo Finance"""
    try:
        # Tambah .JK untuk saham Indonesia
        if not ticker.endswith('.JK') and not '.' in ticker:
            ticker_yf = ticker + '.JK'
        else:
            ticker_yf = ticker

        stock = yf.Ticker(ticker_yf)
        df = stock.history(period=period)

        if df.empty:
            # Coba tanpa .JK (untuk saham US)
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

        return df, stock
    except Exception as e:
        st.error(f"Error mengambil data: {e}")
        return pd.DataFrame(), None

@st.cache_data(ttl=300)
def get_stock_info(ticker: str) -> dict:
    """Ambil info fundamental saham"""
    try:
        if not ticker.endswith('.JK') and not '.' in ticker:
            ticker_yf = ticker + '.JK'
        else:
            ticker_yf = ticker

        stock = yf.Ticker(ticker_yf)
        info = stock.info

        if not info or info.get('regularMarketPrice') is None:
            stock = yf.Ticker(ticker)
            info = stock.info

        return info
    except:
        return {}

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua indikator teknikal"""
    if df.empty or len(df) < 20:
        return df

    # Moving Averages
    df['MA5']   = df['Close'].rolling(window=5).mean()
    df['MA10']  = df['Close'].rolling(window=10).mean()
    df['MA20']  = df['Close'].rolling(window=20).mean()
    df['MA50']  = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # EMA
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # RSI
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    except:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic
    try:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
    except:
        pass

    # Volume indicators
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA20']

    # ATR (Average True Range)
    try:
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    except:
        pass

    # Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

    return df

def generate_recommendation(df: pd.DataFrame, info: dict) -> dict:
    """Generate rekomendasi berdasarkan analisis teknikal"""
    if df.empty or len(df) < 20:
        return {"recommendation": "INSUFFICIENT DATA", "score": 0, "signals": []}

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    signals = []
    score   = 0  # Positif = bullish, negatif = bearish

    # ── RSI ──────────────────────────────────────────────────
    if 'RSI' in df.columns and not pd.isna(latest['RSI']):
        rsi = latest['RSI']
        if rsi < 30:
            signals.append({"indicator": "RSI", "signal": "BUY",
                            "detail": f"RSI {rsi:.1f} - Oversold (< 30)", "weight": 2})
            score += 2
        elif rsi < 40:
            signals.append({"indicator": "RSI", "signal": "BUY",
                            "detail": f"RSI {rsi:.1f} - Mendekati Oversold", "weight": 1})
            score += 1
        elif rsi > 70:
            signals.append({"indicator": "RSI", "signal": "SELL",
                            "detail": f"RSI {rsi:.1f} - Overbought (> 70)", "weight": 2})
            score -= 2
        elif rsi > 60:
            signals.append({"indicator": "RSI", "signal": "SELL",
                            "detail": f"RSI {rsi:.1f} - Mendekati Overbought", "weight": 1})
            score -= 1
        else:
            signals.append({"indicator": "RSI", "signal": "HOLD",
                            "detail": f"RSI {rsi:.1f} - Netral (30-70)", "weight": 0})

    # ── MACD ─────────────────────────────────────────────────
    if 'MACD' in df.columns and not pd.isna(latest['MACD']):
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals.append({"indicator": "MACD", "signal": "BUY",
                            "detail": "MACD Golden Cross - Bullish Crossover", "weight": 3})
            score += 3
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals.append({"indicator": "MACD", "signal": "SELL",
                            "detail": "MACD Death Cross - Bearish Crossover", "weight": 3})
            score -= 3
        elif latest['MACD'] > latest['MACD_signal']:
            signals.append({"indicator": "MACD", "signal": "BUY",
                            "detail": "MACD di atas Signal Line - Bullish", "weight": 1})
            score += 1
        else:
            signals.append({"indicator": "MACD", "signal": "SELL",
                            "detail": "MACD di bawah Signal Line - Bearish", "weight": 1})
            score -= 1

    # ── Moving Average ────────────────────────────────────────
    if 'MA20' in df.columns and not pd.isna(latest['MA20']):
        if latest['Close'] > latest['MA20']:
            signals.append({"indicator": "MA20", "signal": "BUY",
                            "detail": f"Harga di atas MA20 ({latest['MA20']:.0f})", "weight": 1})
            score += 1
        else:
            signals.append({"indicator": "MA20", "signal": "SELL",
                            "detail": f"Harga di bawah MA20 ({latest['MA20']:.0f})", "weight": 1})
            score -= 1

    if 'MA50' in df.columns and not pd.isna(latest['MA50']):
        if latest['Close'] > latest['MA50']:
            signals.append({"indicator": "MA50", "signal": "BUY",
                            "detail": f"Harga di atas MA50 ({latest['MA50']:.0f})", "weight": 1})
            score += 1
        else:
            signals.append({"indicator": "MA50", "signal": "SELL",
                            "detail": f"Harga di bawah MA50 ({latest['MA50']:.0f})", "weight": 1})
            score -= 1

    # ── Golden/Death Cross MA50 vs MA200 ─────────────────────
    if 'MA50' in df.columns and 'MA200' in df.columns:
        if not pd.isna(latest['MA50']) and not pd.isna(latest['MA200']):
            if latest['MA50'] > latest['MA200']:
                signals.append({"indicator": "MA50/200", "signal": "BUY",
                                "detail": "Golden Cross: MA50 > MA200 - Bullish Jangka Panjang", "weight": 2})
                score += 2
            else:
                signals.append({"indicator": "MA50/200", "signal": "SELL",
                                "detail": "Death Cross: MA50 < MA200 - Bearish Jangka Panjang", "weight": 2})
                score -= 2

    # ── Bollinger Bands ───────────────────────────────────────
    if 'BB_upper' in df.columns and not pd.isna(latest['BB_upper']):
        if latest['Close'] < latest['BB_lower']:
            signals.append({"indicator": "Bollinger Bands", "signal": "BUY",
                            "detail": "Harga di bawah BB Lower - Potensi Rebound", "weight": 2})
            score += 2
        elif latest['Close'] > latest['BB_upper']:
            signals.append({"indicator": "Bollinger Bands", "signal": "SELL",
                            "detail": "Harga di atas BB Upper - Potensi Koreksi", "weight": 2})
            score -= 2
        else:
            bb_pos = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            signals.append({"indicator": "Bollinger Bands", "signal": "HOLD",
                            "detail": f"Harga dalam BB ({bb_pos*100:.0f}% dari lower)", "weight": 0})

    # ── Volume ────────────────────────────────────────────────
    if 'Volume_ratio' in df.columns and not pd.isna(latest['Volume_ratio']):
        if latest['Volume_ratio'] > 1.5 and latest['Daily_Return'] > 0:
            signals.append({"indicator": "Volume", "signal": "BUY",
                            "detail": f"Volume tinggi ({latest['Volume_ratio']:.1f}x) dengan harga naik", "weight": 1})
            score += 1
        elif latest['Volume_ratio'] > 1.5 and latest['Daily_Return'] < 0:
            signals.append({"indicator": "Volume", "signal": "SELL",
                            "detail": f"Volume tinggi ({latest['Volume_ratio']:.1f}x) dengan harga turun", "weight": 1})
            score -= 1

    # ── Final Recommendation ──────────────────────────────────
    if score >= 4:
        recommendation = "STRONG BUY"
    elif score >= 2:
        recommendation = "BUY"
    elif score <= -4:
        recommendation = "STRONG SELL"
    elif score <= -2:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "recommendation": recommendation,
        "score": score,
        "signals": signals,
        "max_score": 12
    }

def predict_price(df: pd.DataFrame, days: int = 30) -> dict:
    """Prediksi harga menggunakan Linear Regression + trend"""
    if df.empty or len(df) < 30:
        return {}

    try:
        close_prices = df['Close'].values
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        # Linear Regression
        model = LinearRegression()
        model.fit(X, y)

        # Prediksi ke depan
        future_X = np.arange(len(close_prices), len(close_prices) + days).reshape(-1, 1)
        predictions = model.predict(future_X)

        # Hitung volatilitas untuk confidence interval
        residuals = y - model.predict(X)
        std_residuals = np.std(residuals)

        # Trend analysis
        trend_slope = model.coef_[0]
        current_price = close_prices[-1]

        # 30-day prediction
        pred_30d = predictions[-1]
        pred_7d  = predictions[6] if days >= 7 else predictions[-1]
        pred_14d = predictions[13] if days >= 14 else predictions[-1]

        # Confidence interval (95%)
        ci_upper = pred_30d + 1.96 * std_residuals
        ci_lower = pred_30d - 1.96 * std_residuals

        # Future dates
        last_date    = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')

        return {
            "predictions": predictions,
            "future_dates": future_dates,
            "pred_7d": pred_7d,
            "pred_14d": pred_14d,
            "pred_30d": pred_30d,
            "ci_upper": ci_upper,
            "ci_lower": ci_lower,
            "trend_slope": trend_slope,
            "current_price": current_price,
            "change_7d_pct": ((pred_7d - current_price) / current_price) * 100,
            "change_14d_pct": ((pred_14d - current_price) / current_price) * 100,
            "change_30d_pct": ((pred_30d - current_price) / current_price) * 100,
            "std_residuals": std_residuals
        }
    except Exception as e:
        st.warning(f"Prediksi error: {e}")
        return {}

def format_number(num, prefix="Rp "):
    """Format angka besar"""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"
    if abs(num) >= 1e12:
        return f"{prefix}{num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"{prefix}{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.2f}M"
    else:
        return f"{prefix}{num:,.0f}"

def get_signal_color(signal: str) -> str:
    colors = {"BUY": "🟢", "STRONG BUY": "🟢🟢", "SELL": "🔴",
              "STRONG SELL": "🔴🔴", "HOLD": "🟡"}
    return colors.get(signal, "⚪")

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<div class="main-header">📈 Stock Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
        st.title("⚙️ Settings")

        st.subheader("🔍 Pilih Saham")

        # Watchlist populer
        popular_stocks = {
            "🇮🇩 IDX Popular": ["BBCA", "BBRI", "TLKM", "ASII", "BMRI",
                                  "GOTO", "BYAN", "UNVR", "ICBP", "KLBF"],
            "🇺🇸 US Stocks":   ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA",
                                  "AMZN", "META", "NFLX", "AMD", "BABA"],
            "₿ Crypto (via YF)": ["BTC-USD", "ETH-USD", "BNB-USD"]
        }

        market = st.selectbox("Market", list(popular_stocks.keys()))

        # Input manual atau pilih dari list
        input_mode = st.radio("Mode Input", ["Pilih dari list", "Ketik manual"])

        if input_mode == "Pilih dari list":
            ticker_input = st.selectbox("Pilih Saham", popular_stocks[market])
        else:
            ticker_input = st.text_input("Ticker Symbol", value="BBCA",
                                          help="Contoh: BBCA (IDX), AAPL (US). Saham IDX otomatis ditambah .JK")

        # Period
        period_map = {
            "1 Bulan": "1mo", "3 Bulan": "3mo", "6 Bulan": "6mo",
            "1 Tahun": "1y",  "2 Tahun": "2y",  "5 Tahun": "5y"
        }
        period_label  = st.selectbox("Periode Data", list(period_map.keys()), index=3)
        selected_period = period_map[period_label]

        # Prediction days
        pred_days = st.slider("Hari Prediksi ke Depan", 7, 90, 30)

        # Analyze button
        analyze_btn = st.button("🔍 Analisis Saham", type="primary", use_container_width=True)

        st.markdown("---")

        # Watchlist
        st.subheader("📋 Watchlist Cepat")
        watchlist = st.multiselect(
            "Tambah ke Watchlist",
            ["BBCA", "BBRI", "TLKM", "ASII", "BMRI", "AAPL", "GOOGL", "TSLA"],
            default=["BBCA", "BBRI", "TLKM"]
        )

        st.markdown("---")
        st.caption(f"🕐 Last update: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption("⚠️ Data dari Yahoo Finance. Bukan saran investasi.")

    # ── Main Content ──────────────────────────────────────────
    ticker = ticker_input.strip().upper()

    if not ticker:
        st.info("👈 Masukkan ticker saham di sidebar untuk memulai analisis")
        _show_market_overview(watchlist)
        return

    # Load data
    with st.spinner(f"⏳ Mengambil data {ticker}..."):
        df_raw, stock_obj = get_stock_data(ticker, selected_period)
        info = get_stock_info(ticker)

    if df_raw.empty:
        st.error(f"❌ Data untuk {ticker} tidak ditemukan. Cek kembali ticker symbol.")
        return

    # Hitung indikator
    df = calculate_technical_indicators(df_raw.copy())

    # Generate rekomendasi & prediksi
    rec_data  = generate_recommendation(df, info)
    pred_data = predict_price(df, pred_days)

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    price_change     = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100

    # ── Stock Header ──────────────────────────────────────────
    col_title, col_rec = st.columns([2, 1])

    with col_title:
        company_name = info.get('longName', ticker)
        sector       = info.get('sector', 'N/A')
        industry     = info.get('industry', 'N/A')

        st.markdown(f"## {company_name}")
        st.markdown(f"**{ticker}** | {sector} | {industry}")

        price_color = "green" if price_change >= 0 else "red"
        arrow       = "▲" if price_change >= 0 else "▼"

        st.markdown(
            f"### <span style='color:{price_color}'>"
            f"{latest['Close']:,.0f} "
            f"{arrow} {abs(price_change):,.0f} ({abs(price_change_pct):.2f}%)"
            f"</span>",
            unsafe_allow_html=True
        )

    with col_rec:
        rec = rec_data['recommendation']
        css_class = {
            "STRONG BUY": "rec-buy", "BUY": "rec-buy",
            "STRONG SELL": "rec-sell", "SELL": "rec-sell",
            "HOLD": "rec-hold"
        }.get(rec, "rec-hold")

        emoji = {"STRONG BUY": "🚀", "BUY": "📈", "STRONG SELL": "💥",
                 "SELL": "📉", "HOLD": "⏸️"}.get(rec, "❓")

        st.markdown(
            f'<div class="{css_class}">{emoji} {rec}<br>'
            f'<small>Score: {rec_data["score"]}/12</small></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Key Metrics ───────────────────────────────────────────
    st.subheader("📊 Key Metrics")

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric("💰 Harga", f"{latest['Close']:,.0f}",
                  f"{price_change_pct:+.2f}%")
    with m2:
        st.metric("📈 High (Hari)", f"{latest['High']:,.0f}")
    with m3:
        st.metric("📉 Low (Hari)", f"{latest['Low']:,.0f}")
    with m4:
        vol = latest['Volume']
        st.metric("📦 Volume", f"{vol/1e6:.1f}M" if vol > 1e6 else f"{vol:,.0f}")
    with m5:
        rsi_val = latest.get('RSI', np.nan)
        rsi_str = f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "N/A"
        rsi_delta = "Oversold" if not pd.isna(rsi_val) and rsi_val < 30 else \
                    "Overbought" if not pd.isna(rsi_val) and rsi_val > 70 else "Normal"
        st.metric("📡 RSI (14)", rsi_str, rsi_delta)
    with m6:
        ma20_val = latest.get('MA20', np.nan)
        if not pd.isna(ma20_val):
            ma20_diff = ((latest['Close'] - ma20_val) / ma20_val) * 100
            st.metric("📏 vs MA20", f"{ma20_val:,.0f}", f"{ma20_diff:+.2f}%")
        else:
            st.metric("📏 MA20", "N/A")

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart & Teknikal",
        "🎯 Sinyal & Rekomendasi",
        "🔮 Prediksi Harga",
        "📋 Fundamental",
        "📊 Statistik"
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: CHART
    # ════════════════════════════════════════════════════════
    with tab1:
        st.subheader("📈 Candlestick Chart + Indikator")

        chart_options = st.multiselect(
            "Tampilkan Indikator",
            ["MA20", "MA50", "MA200", "Bollinger Bands", "Volume"],
            default=["MA20", "MA50", "Bollinger Bands", "Volume"]
        )

        # Buat subplot
        rows = 3 if "Volume" in chart_options else 2
        row_heights = [0.6, 0.2, 0.2] if rows == 3 else [0.7, 0.3]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=("Price", "MACD", "Volume" if rows == 3 else "MACD")
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'],   close=df['Close'],
            name="Price",
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444'
        ), row=1, col=1)

        # Moving Averages
        ma_colors = {"MA20": "#FFB300", "MA50": "#2196F3", "MA200": "#E91E63"}
        for ma in ["MA20", "MA50", "MA200"]:
            if ma in chart_options and ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma],
                    name=ma, line=dict(color=ma_colors[ma], width=1.5)
                ), row=1, col=1)

        # Bollinger Bands
        if "Bollinger Bands" in chart_options and 'BB_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_upper'],
                name="BB Upper", line=dict(color='rgba(100,100,255,0.5)', dash='dash'),
                showlegend=True
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_lower'],
                name="BB Lower", line=dict(color='rgba(100,100,255,0.5)', dash='dash'),
                fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
                showlegend=True
            ), row=1, col=1)

        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                name="MACD", line=dict(color='#2196F3', width=1.5)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_signal'],
                name="Signal", line=dict(color='#FF9800', width=1.5)
            ), row=2, col=1)
            colors_hist = ['#00C851' if v >= 0 else '#FF4444' for v in df['MACD_hist'].fillna(0)]
            fig.add_trace(go.Bar(
                x=df.index, y=df['MACD_hist'],
                name="Histogram", marker_color=colors_hist
            ), row=2, col=1)

        # Volume
        if "Volume" in chart_options and rows == 3:
            vol_colors = ['#00C851' if df['Close'].iloc[i] >= df['Open'].iloc[i]
                          else '#FF4444' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name="Volume", marker_color=vol_colors, showlegend=False
            ), row=3, col=1)
            if 'Volume_MA20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Volume_MA20'],
                    name="Vol MA20", line=dict(color='yellow', width=1)
                ), row=3, col=1)

        fig.update_layout(
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # RSI Chart
        if 'RSI' in df.columns:
            st.subheader("📡 RSI (14)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df['RSI'],
                name="RSI", line=dict(color='#9C27B0', width=2),
                fill='tozeroy', fillcolor='rgba(156,39,176,0.1)'
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                               annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                               annotation_text="Oversold (30)")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_rsi.update_layout(
                height=250, template="plotly_dark",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 2: SINYAL & REKOMENDASI
    # ════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🎯 Analisis Sinyal Teknikal")

        col_gauge, col_signals = st.columns([1, 2])

        with col_gauge:
            # Gauge chart untuk score
            score     = rec_data['score']
            max_score = rec_data['max_score']

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Bullish Score", 'font': {'size': 20}},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-max_score, max_score], 'tickwidth': 1},
                    'bar': {'color': "#00C851" if score > 0 else "#FF4444"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-max_score, -4], 'color': '#FF1744'},
                        {'range': [-4, -2],          'color': '#FF7043'},
                        {'range': [-2, 2],            'color': '#FFC107'},
                        {'range': [2, 4],             'color': '#66BB6A'},
                        {'range': [4, max_score],     'color': '#00C851'},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig_gauge.update_layout(
                height=300, template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Rekomendasi besar
            rec = rec_data['recommendation']
            emoji = {"STRONG BUY": "🚀", "BUY": "📈", "STRONG SELL": "💥",
                     "SELL": "📉", "HOLD": "⏸️"}.get(rec, "❓")
            css_class = "rec-buy" if "BUY" in rec else \
                        "rec-sell" if "SELL" in rec else "rec-hold"
            st.markdown(
                f'<div class="{css_class}" style="margin-top:1rem">'
                f'{emoji} {rec}</div>',
                unsafe_allow_html=True
            )

        with col_signals:
            st.subheader("📋 Detail Sinyal")

            signals = rec_data.get('signals', [])

            buy_signals  = [s for s in signals if s['signal'] == 'BUY']
            sell_signals = [s for s in signals if s['signal'] == 'SELL']
            hold_signals = [s for s in signals if s['signal'] == 'HOLD']

            if buy_signals:
                st.markdown("**🟢 Sinyal BUY:**")
                for s in buy_signals:
                    st.markdown(
                        f'<div class="info-box">✅ <b>{s["indicator"]}</b>: {s["detail"]}</div>',
                        unsafe_allow_html=True
                    )

            if sell_signals:
                st.markdown("**🔴 Sinyal SELL:**")
                for s in sell_signals:
                    st.markdown(
                        f'<div class="info-box" style="border-left: 3px solid #FF4444;">❌ <b>{s["indicator"]}</b>: {s["detail"]}</div>',
                        unsafe_allow_html=True
                    )

            if hold_signals:
                st.markdown("**🟡 Sinyal HOLD/NETRAL:**")
                for s in hold_signals:
                    st.markdown(
                        f'<div class="info-box" style="border-left: 3px solid #FFB300;">⏸️ <b>{s["indicator"]}</b>: {s["detail"]}</div>',
                        unsafe_allow_html=True
                    )

        # Summary table
        st.markdown("---")
        st.subheader("📊 Ringkasan Indikator")

        indicator_data = []
        for s in signals:
            indicator_data.append({
                "Indikator": s['indicator'],
                "Sinyal": s['signal'],
                "Detail": s['detail'],
                "Bobot": s['weight']
            })

        if indicator_data:
            df_signals = pd.DataFrame(indicator_data)

            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #1B5E20; color: white'
                elif val == 'SELL':
                    return 'background-color: #B71C1C; color: white'
                else:
                    return 'background-color: #E65100; color: white'

            styled_df = df_signals.style.applymap(color_signal, subset=['Sinyal'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Support & Resistance
        st.markdown("---")
        st.subheader("🎯 Support & Resistance")

        col_sr1, col_sr2, col_sr3 = st.columns(3)

        # Hitung S&R sederhana
        recent_data = df.tail(60)
        resistance  = recent_data['High'].max()
        support     = recent_data['Low'].min()
        pivot       = (recent_data['High'].iloc[-1] + recent_data['Low'].iloc[-1] + recent_data['Close'].iloc[-1]) / 3

        with col_sr1:
            st.metric("🔴 Resistance", f"{resistance:,.0f}",
                      f"{((resistance - latest['Close'])/latest['Close'])*100:+.2f}%")
        with col_sr2:
            st.metric("🟡 Pivot Point", f"{pivot:,.0f}",
                      f"{((pivot - latest['Close'])/latest['Close'])*100:+.2f}%")
        with col_sr3:
            st.metric("🟢 Support", f"{support:,.0f}",
                      f"{((support - latest['Close'])/latest['Close'])*100:+.2f}%")

    # ════════════════════════════════════════════════════════
    # TAB 3: PREDIKSI
    # ════════════════════════════════════════════════════════
    with tab3:
        st.subheader(f"🔮 Prediksi Harga {pred_days} Hari ke Depan")

        st.warning("⚠️ **Disclaimer**: Prediksi ini menggunakan Linear Regression berdasarkan data historis. "
                   "Ini BUKAN jaminan pergerakan harga. Selalu lakukan riset mandiri sebelum berinvestasi.")

        if pred_data:
            # Prediction metrics
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)

            with col_p1:
                st.metric("📍 Harga Sekarang", f"{pred_data['current_price']:,.0f}")
            with col_p2:
                st.metric(
                    "📅 Prediksi 7 Hari",
                    f"{pred_data['pred_7d']:,.0f}",
                    f"{pred_data['change_7d_pct']:+.2f}%"
                )
            with col_p3:
                st.metric(
                    "📅 Prediksi 14 Hari",
                    f"{pred_data['pred_14d']:,.0f}",
                    f"{pred_data['change_14d_pct']:+.2f}%"
                )
            with col_p4:
                st.metric(
                    f"📅 Prediksi {pred_days} Hari",
                    f"{pred_data['pred_30d']:,.0f}",
                    f"{pred_data['change_30d_pct']:+.2f}%"
                )

            # Trend direction
            slope = pred_data['trend_slope']
            if slope > 0:
                st.success(f"📈 **Trend: UPTREND** - Slope: +{slope:.2f} per hari")
            else:
                st.error(f"📉 **Trend: DOWNTREND** - Slope: {slope:.2f} per hari")

            # Prediction chart
            fig_pred = go.Figure()

            # Historical
            hist_days = min(90, len(df))
            fig_pred.add_trace(go.Scatter(
                x=df.index[-hist_days:],
                y=df['Close'].iloc[-hist_days:],
                name="Harga Historis",
                line=dict(color='#2196F3', width=2)
            ))

            # Prediction
            fig_pred.add_trace(go.Scatter(
                x=pred_data['future_dates'],
                y=pred_data['predictions'],
                name="Prediksi",
                line=dict(color='#FF9800', width=2, dash='dash')
            ))

            # Confidence interval
            fig_pred.add_trace(go.Scatter(
                x=list(pred_data['future_dates']) + list(pred_data['future_dates'])[::-1],
                y=list(pred_data['predictions'] + 1.96 * pred_data['std_residuals']) +
                  list(pred_data['predictions'] - 1.96 * pred_data['std_residuals'])[::-1],
                fill='toself',
                fillcolor='rgba(255,152,0,0.15)',
                line=dict(color='rgba(255,152,0,0)'),
                name="Confidence Interval 95%"
            ))

            # Vertical line (today)
            fig_pred.add_vline(
                x=df.index[-1], line_dash="dot",
                line_color="white", annotation_text="Hari Ini"
            )

            fig_pred.update_layout(
                height=450, template="plotly_dark",
                title=f"Prediksi Harga {ticker} - {pred_days} Hari ke Depan",
                xaxis_title="Tanggal",
                yaxis_title="Harga",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_pred, use_container_width=True)

            # Prediction table
            st.subheader("📋 Tabel Prediksi")
            pred_table = pd.DataFrame({
                'Tanggal': pred_data['future_dates'][:pred_days:5],
                'Prediksi Harga': pred_data['predictions'][:pred_days:5],
                'Batas Atas (95%)': pred_data['predictions'][:pred_days:5] + 1.96 * pred_data['std_residuals'],
                'Batas Bawah (95%)': pred_data['predictions'][:pred_days:5] - 1.96 * pred_data['std_residuals'],
            })
            pred_table['Prediksi Harga'] = pred_table['Prediksi Harga'].map('{:,.0f}'.format)
            pred_table['Batas Atas (95%)'] = pred_table['Batas Atas (95%)'].map('{:,.0f}'.format)
            pred_table['Batas Bawah (95%)'] = pred_table['Batas Bawah (95%)'].map('{:,.0f}'.format)
            pred_table['Tanggal'] = pred_table['Tanggal'].dt.strftime('%d %b %Y')

            st.dataframe(pred_table, use_container_width=True, hide_index=True)
        else:
            st.error("❌ Tidak cukup data untuk membuat prediksi. Butuh minimal 30 hari data.")

    # ════════════════════════════════════════════════════════
    # TAB 4: FUNDAMENTAL
    # ════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📋 Data Fundamental")

        if info:
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                st.markdown("**🏢 Informasi Perusahaan**")
                company_info = {
                    "Nama": info.get('longName', 'N/A'),
                    "Sektor": info.get('sector', 'N/A'),
                    "Industri": info.get('industry', 'N/A'),
                    "Negara": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Karyawan": f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                for k, v in company_info.items():
                    st.markdown(
                        f'<div class="info-box"><b>{k}:</b> {v}</div>',
                        unsafe_allow_html=True
                    )

            with col_f2:
                st.markdown("**💰 Valuasi**")

                market_cap = info.get('marketCap')
                pe_ratio   = info.get('trailingPE')
                pb_ratio   = info.get('priceToBook')
                eps        = info.get('trailingEps')
                div_yield  = info.get('dividendYield')
                beta       = info.get('beta')

                valuation = {
                    "Market Cap": format_number(market_cap, ""),
                    "P/E Ratio": f"{pe_ratio:.2f}x" if pe_ratio else 'N/A',
                    "P/B Ratio": f"{pb_ratio:.2f}x" if pb_ratio else 'N/A',
                    "EPS (TTM)": f"{eps:.2f}" if eps else 'N/A',
                    "Dividend Yield": f"{div_yield*100:.2f}%" if div_yield else 'N/A',
                    "Beta": f"{beta:.2f}" if beta else 'N/A',
                    "52W High": f"{info.get('fiftyTwoWeekHigh', 0):,.0f}",
                    "52W Low": f"{info.get('fiftyTwoWeekLow', 0):,.0f}",
                }

                for k, v in valuation.items():
                    st.markdown(
                        f'<div class="info-box"><b>{k}:</b> {v}</div>',
                        unsafe_allow_html=True
                    )

            # Business summary
            summary = info.get('longBusinessSummary', '')
            if summary:
                st.markdown("---")
                st.subheader("📝 Deskripsi Bisnis")
                with st.expander("Lihat Deskripsi Lengkap"):
                    st.write(summary)

            # Financial metrics
            st.markdown("---")
            st.subheader("📊 Metrik Keuangan")

            col_fin1, col_fin2, col_fin3 = st.columns(3)

            with col_fin1:
                revenue = info.get('totalRevenue')
                st.metric("💵 Revenue", format_number(revenue, ""))

                gross_profit = info.get('grossProfits')
                st.metric("💹 Gross Profit", format_number(gross_profit, ""))

            with col_fin2:
                ebitda = info.get('ebitda')
                st.metric("📈 EBITDA", format_number(ebitda, ""))

                net_income = info.get('netIncomeToCommon')
                st.metric("💰 Net Income", format_number(net_income, ""))

            with col_fin3:
                total_debt = info.get('totalDebt')
                st.metric("💳 Total Debt", format_number(total_debt, ""))

                free_cash = info.get('freeCashflow')
                st.metric("💸 Free Cash Flow", format_number(free_cash, ""))
        else:
            st.warning("⚠️ Data fundamental tidak tersedia untuk saham ini.")

    # ════════════════════════════════════════════════════════
    # TAB 5: STATISTIK
    # ════════════════════════════════════════════════════════
    with tab5:
        st.subheader("📊 Statistik & Analisis Lanjutan")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            # Return distribution
            st.markdown("**📊 Distribusi Return Harian**")
            returns = df['Daily_Return'].dropna() * 100

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=returns, nbinsx=50,
                name="Return Distribution",
                marker_color='#2196F3',
                opacity=0.7
            ))
            fig_dist.add_vline(x=returns.mean(), line_dash="dash",
                                line_color="yellow", annotation_text=f"Mean: {returns.mean():.2f}%")
            fig_dist.update_layout(
                height=300, template="plotly_dark",
                title="Distribusi Return Harian (%)",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_s2:
            # Cumulative return
            st.markdown("**📈 Cumulative Return**")
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=df.index,
                y=df['Cumulative_Return'] * 100,
                name="Cumulative Return",
                line=dict(color='#00C851', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,200,81,0.1)'
            ))
            fig_cum.add_hline(y=0, line_dash="dash", line_color="white")
            fig_cum.update_layout(
                height=300, template="plotly_dark",
                title="Cumulative Return (%)",
                yaxis_title="%",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # Statistics table
        st.markdown("---")
        st.subheader("📋 Statistik Deskriptif")

        stats_data = {
            "Metrik": [
                "Return Harian Rata-rata", "Volatilitas (Std Dev)",
                "Return Terbaik", "Return Terburuk",
                "Sharpe Ratio (approx)", "Max Drawdown",
                "Win Rate (hari naik)", "Total Return Periode"
            ],
            "Nilai": [
                f"{returns.mean():.3f}%",
                f"{returns.std():.3f}%",
                f"{returns.max():.3f}%",
                f"{returns.min():.3f}%",
                f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}",
                f"{((df['Close'] / df['Close'].cummax() - 1).min() * 100):.2f}%",
                f"{(returns > 0).sum() / len(returns) * 100:.1f}%",
                f"{df['Cumulative_Return'].iloc[-1] * 100:.2f}%"
            ]
        }

        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        # Monthly returns heatmap
        st.markdown("---")
        st.subheader("🗓️ Return Bulanan")

        try:
            monthly_returns = df['Daily_Return'].resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100

            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.strftime('%b'),
                'Return': monthly_returns.values
            })

            pivot_monthly = monthly_df.pivot_table(
                values='Return', index='Year', columns='Month', aggfunc='first'
            )

            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_monthly = pivot_monthly.reindex(
                columns=[m for m in month_order if m in pivot_monthly.columns]
            )

            fig_heat = px.imshow(
                pivot_monthly,
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title="Monthly Returns Heatmap (%)",
                text_auto='.1f'
            )
            fig_heat.update_layout(
                height=300, template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.info(f"Heatmap tidak tersedia: {e}")

    # ── Download Data ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Data")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = df[['Open', 'High', 'Low', 'Close', 'Volume',
                        'MA20', 'MA50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].to_csv()
        st.download_button(
            label="📥 Download Data CSV",
            data=csv_data,
            file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_dl2:
        # Report summary
        report = f"""
LAPORAN ANALISIS SAHAM - {ticker}
Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*50}

INFORMASI SAHAM:
- Nama: {info.get('longName', ticker)}
- Sektor: {info.get('sector', 'N/A')}
- Harga Terakhir: {latest['Close']:,.0f}
- Perubahan: {price_change:+,.0f} ({price_change_pct:+.2f}%)

REKOMENDASI: {rec_data['recommendation']}
Score: {rec_data['score']}/12

INDIKATOR TEKNIKAL:
- RSI (14): {latest.get('RSI', 'N/A'):.1f if not pd.isna(latest.get('RSI', float('nan'))) else 'N/A'}
- MACD: {latest.get('MACD', 'N/A'):.2f if not pd.isna(latest.get('MACD', float('nan'))) else 'N/A'}
- MA20: {latest.get('MA20', 'N/A'):.0f if not pd.isna(latest.get('MA20', float('nan'))) else 'N/A'}
- MA50: {latest.get('MA50', 'N/A'):.0f if not pd.isna(latest.get('MA50', float('nan'))) else 'N/A'}

PREDIKSI:
- 7 Hari: {pred_data.get('pred_7d', 0):,.0f} ({pred_data.get('change_7d_pct', 0):+.2f}%)
- 14 Hari: {pred_data.get('pred_14d', 0):,.0f} ({pred_data.get('change_14d_pct', 0):+.2f}%)
- 30 Hari: {pred_data.get('pred_30d', 0):,.0f} ({pred_data.get('change_30d_pct', 0):+.2f}%)

SINYAL:
"""
        for s in rec_data.get('signals', []):
            report += f"- [{s['signal']}] {s['indicator']}: {s['detail']}\n"

        report += "\n⚠️ DISCLAIMER: Ini bukan saran investasi. Lakukan riset mandiri."

        st.download_button(
            label="📄 Download Laporan TXT",
            data=report,
            file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================
# MARKET OVERVIEW (Halaman awal)
# ============================================================
def _show_market_overview(watchlist: list):
    st.subheader("📊 Market Overview - Watchlist")

    if not watchlist:
        st.info("Tambahkan saham ke watchlist di sidebar")
        return

    cols = st.columns(len(watchlist))

    for i, ticker in enumerate(watchlist):
        with cols[i]:
            with st.spinner(f"Loading {ticker}..."):
                try:
                    ticker_yf = ticker + '.JK' if '.' not in ticker else ticker
                    stock = yf.Ticker(ticker_yf)
                    hist  = stock.history(period="5d")

                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev    = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change  = ((current - prev) / prev) * 100

                        st.metric(
                            label=f"📈 {ticker}",
                            value=f"{current:,.0f}",
                            delta=f"{change:+.2f}%"
                        )
                    else:
                        st.metric(ticker, "N/A")
                except Exception:
                    st.metric(ticker, "Error")

    st.markdown("---")
    st.info("👈 Pilih saham di sidebar dan klik **Analisis Saham** untuk melihat analisis lengkap")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
