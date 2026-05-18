# app.py - Single File Version (Streamlit Cloud Compatible)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
import re
import os
import google.generativeai as genai
from datetime import datetime

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇩 Analisis Saham Indonesia AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🇮🇩 Analisis Saham Indonesia</h1>
    <p>Powered by Gemini 2.5 Flash AI • Real-time Technical & Fundamental Analysis</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_indonesian_stocks():
    return {
        "BBCA - Bank Central Asia": "BBCA.JK",
        "BBRI - Bank Rakyat Indonesia": "BBRI.JK",
        "BMRI - Bank Mandiri": "BMRI.JK",
        "TLKM - Telkom Indonesia": "TLKM.JK",
        "ASII - Astra International": "ASII.JK",
        "GOTO - GoTo Gojek Tokopedia": "GOTO.JK",
        "BYAN - Bayan Resources": "BYAN.JK",
        "ADRO - Adaro Energy": "ADRO.JK",
        "UNVR - Unilever Indonesia": "UNVR.JK",
        "ICBP - Indofood CBP": "ICBP.JK",
        "INDF - Indofood Sukses Makmur": "INDF.JK",
        "KLBF - Kalbe Farma": "KLBF.JK",
        "ANTM - Aneka Tambang": "ANTM.JK",
        "PTBA - Bukit Asam": "PTBA.JK",
        "SMGR - Semen Indonesia": "SMGR.JK",
        "PGAS - Perusahaan Gas Negara": "PGAS.JK",
        "JSMR - Jasa Marga": "JSMR.JK",
        "EXCL - XL Axiata": "EXCL.JK",
        "INCO - Vale Indonesia": "INCO.JK",
        "MDKA - Merdeka Copper Gold": "MDKA.JK",
    }


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        raise Exception(f"Gagal mengambil data saham: {str(e)}")


@st.cache_data(ttl=300)
def fetch_stock_info(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Nama Perusahaan": info.get("longName", "N/A"),
            "Sektor": info.get("sector", "N/A"),
            "Industri": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "P/B Ratio": info.get("priceToBook", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Avg Volume": info.get("averageVolume", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "ROA": info.get("returnOnAssets", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Current Ratio": info.get("currentRatio", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Earnings Growth": info.get("earningsGrowth", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Operating Margin": info.get("operatingMargins", "N/A"),
        }
    except Exception as e:
        return {"Error": str(e)}


def get_current_price(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")
        if len(hist) >= 2:
            current = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            change = current - prev
            change_pct = (change / prev) * 100
        elif len(hist) == 1:
            current = hist["Close"].iloc[-1]
            change = 0
            change_pct = 0
        else:
            return {"current_price": 0, "change": 0, "change_pct": 0, "volume": 0}

        return {
            "current_price": round(float(current), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "volume": int(hist["Volume"].iloc[-1]) if len(hist) > 0 else 0,
        }
    except Exception as e:
        return {"current_price": 0, "change": 0, "change_pct": 0, "volume": 0}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Mid"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    )

    # Stochastic
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # ATR
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["ATR"] = df["TR"].rolling(window=14).mean()

    # OBV
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # Volume MA
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    # Williams %R
    df["Williams_R"] = -100 * (high_14 - df["Close"]) / (high_14 - low_14)

    # CCI
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_deviation = typical_price.rolling(window=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    df["CCI"] = (typical_price - typical_price.rolling(window=20).mean()) / (
        0.015 * mean_deviation
    )

    # ROC
    df["ROC"] = df["Close"].pct_change(periods=12) * 100

    return df


def get_indicator_summary(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    signals = {}

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "OVERBOUGHT ⚠️", "bias": "BEARISH"}
    elif rsi < 30:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "OVERSOLD 🟢", "bias": "BULLISH"}
    else:
        signals["RSI"] = {"value": round(rsi, 2), "signal": "NORMAL", "bias": "NEUTRAL"}

    # MACD
    macd_cross = "BULLISH" if latest["MACD"] > latest["MACD_Signal"] else "BEARISH"
    golden_cross = (latest["MACD"] > latest["MACD_Signal"] and
                    prev["MACD"] <= prev["MACD_Signal"])
    dead_cross = (latest["MACD"] < latest["MACD_Signal"] and
                  prev["MACD"] >= prev["MACD_Signal"])
    macd_signal_text = ("GOLDEN CROSS 🚀" if golden_cross
                        else "DEAD CROSS ⚠️" if dead_cross else macd_cross)
    signals["MACD"] = {"value": round(latest["MACD"], 4),
                       "signal": macd_signal_text, "bias": macd_cross}

    # Bollinger Bands
    bb_pos = latest["BB_Position"]
    if bb_pos > 0.9:
        bb_signal, bb_bias = "UPPER BAND (Overbought) ⚠️", "BEARISH"
    elif bb_pos < 0.1:
        bb_signal, bb_bias = "LOWER BAND (Oversold) 🟢", "BULLISH"
    else:
        bb_signal, bb_bias = f"MID BAND ({round(bb_pos*100,1)}%)", "NEUTRAL"
    signals["Bollinger Bands"] = {"value": round(bb_pos, 3),
                                  "signal": bb_signal, "bias": bb_bias}

    # Moving Average
    price = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]
    if price > sma20 and price > sma50:
        ma_signal, ma_bias = "ABOVE MA20 & MA50 🚀", "BULLISH"
    elif price < sma20 and price < sma50:
        ma_signal, ma_bias = "BELOW MA20 & MA50 ⚠️", "BEARISH"
    else:
        ma_signal, ma_bias = "MIXED", "NEUTRAL"
    signals["Moving Average"] = {"value": round(price, 2),
                                 "signal": ma_signal, "bias": ma_bias}

    # Stochastic
    stoch_k = latest["Stoch_K"]
    if stoch_k > 80:
        stoch_signal, stoch_bias = "OVERBOUGHT ⚠️", "BEARISH"
    elif stoch_k < 20:
        stoch_signal, stoch_bias = "OVERSOLD 🟢", "BULLISH"
    else:
        stoch_signal, stoch_bias = "NORMAL", "NEUTRAL"
    signals["Stochastic"] = {"value": round(stoch_k, 2),
                             "signal": stoch_signal, "bias": stoch_bias}

    # Volume
    vol_ratio = latest["Volume_Ratio"]
    if vol_ratio > 1.5:
        vol_signal, vol_bias = f"HIGH VOLUME ({round(vol_ratio,2)}x) 🔥", "STRONG"
    elif vol_ratio < 0.5:
        vol_signal, vol_bias = f"LOW VOLUME ({round(vol_ratio,2)}x)", "WEAK"
    else:
        vol_signal, vol_bias = f"NORMAL ({round(vol_ratio,2)}x)", "NEUTRAL"
    signals["Volume"] = {"value": round(vol_ratio, 2),
                         "signal": vol_signal, "bias": vol_bias}

    bullish_count = sum(1 for s in signals.values() if s["bias"] == "BULLISH")
    bearish_count = sum(1 for s in signals.values() if s["bias"] == "BEARISH")
    signals["_summary"] = {
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "overall": ("BULLISH" if bullish_count > bearish_count
                    else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"),
    }
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: NEWS SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def clean_html(text: str) -> str:
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        clean = soup.get_text()
    except Exception:
        clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


@st.cache_data(ttl=600)
def fetch_google_news(query: str, max_articles: int = 8) -> list:
    articles = []
    try:
        encoded_query = requests.utils.quote(f"{query} saham Indonesia")
        url = (f"https://news.google.com/rss/search?"
               f"q={encoded_query}&hl=id&gl=ID&ceid=ID:id")
        headers = {"User-Agent": "Mozilla/5.0"}
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles]:
            articles.append({
                "source": "Google News",
                "title": entry.get("title", ""),
                "summary": clean_html(entry.get("summary", ""))[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "relevant": True,
            })
    except Exception:
        pass
    return articles


@st.cache_data(ttl=600)
def fetch_stock_news(stock_name: str, ticker: str, max_articles: int = 10) -> list:
    clean_ticker = ticker.replace(".JK", "")
    articles = []

    # Google News untuk saham spesifik
    queries = [clean_ticker, f"{clean_ticker} saham", stock_name.split(" - ")[0]]
    for query in queries[:2]:
        news = fetch_google_news(query, max_articles=5)
        articles.extend(news)

    # RSS feeds Indonesia
    rss_feeds = {
        "Kontan": "https://rss.kontan.co.id/category/investasi",
        "CNBC Indonesia": "https://www.cnbcindonesia.com/rss",
    }
    for source, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                combined = (title + " " + summary).upper()
                if clean_ticker.upper() in combined:
                    articles.append({
                        "source": source,
                        "title": title,
                        "summary": clean_html(summary)[:300],
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "relevant": True,
                    })
        except Exception:
            continue

    # Deduplicate by title
    seen = set()
    unique_articles = []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique_articles.append(a)

    return unique_articles[:max_articles]


@st.cache_data(ttl=600)
def fetch_market_news() -> list:
    articles = []
    queries = ["IHSG", "Bank Indonesia suku bunga", "ekonomi Indonesia"]
    for query in queries:
        news = fetch_google_news(query, max_articles=3)
        articles.extend(news)
    return articles[:10]


def format_news_for_llm(articles: list) -> str:
    if not articles:
        return "Tidak ada berita terkini yang ditemukan."
    formatted = []
    for i, article in enumerate(articles, 1):
        tag = "[RELEVAN]" if article.get("relevant") else "[UMUM]"
        formatted.append(
            f"{i}. {tag} {article['source']}\n"
            f"   Judul: {article['title']}\n"
            f"   Ringkasan: {article['summary']}\n"
            f"   Tanggal: {article['published']}\n"
        )
    return "\n".join(formatted)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LLM ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def build_prompt(ticker, stock_name, current_price, df,
                 fundamental, news_text, market_news) -> str:
    latest = df.iloc[-1]

    price_1w = df["Close"].iloc[-5] if len(df) >= 5 else df["Close"].iloc[0]
    price_1m = df["Close"].iloc[-20] if len(df) >= 20 else df["Close"].iloc[0]
    price_3m = df["Close"].iloc[-60] if len(df) >= 60 else df["Close"].iloc[0]
    current = current_price["current_price"]

    def pct(a, b):
        return ((a - b) / b * 100) if b > 0 else 0

    def fmt(val, is_pct=False):
        if val in ("N/A", None):
            return "N/A"
        try:
            v = float(val)
            if is_pct:
                return f"{v*100:.2f}%"
            if v > 1_000_000_000_000:
                return f"Rp {v/1_000_000_000_000:.2f}T"
            if v > 1_000_000_000:
                return f"Rp {v/1_000_000_000:.2f}B"
            return f"{v:.2f}"
        except Exception:
            return str(val)

    return f"""
Kamu adalah analis saham profesional Indonesia dengan pengalaman 20 tahun di pasar modal IDX.
Analisis data berikut dan berikan rekomendasi investasi yang actionable.

═══════════════════════════════════════════════════════
📊 DATA SAHAM: {stock_name} ({ticker})
Tanggal Analisis: {datetime.now().strftime("%d %B %Y, %H:%M WIB")}
═══════════════════════════════════════════════════════

💰 HARGA & PERFORMA
├── Harga Saat Ini   : Rp {current:,.0f}
├── Perubahan Hari   : {current_price['change']:+.0f} ({current_price['change_pct']:+.2f}%)
├── Volume Hari Ini  : {current_price['volume']:,.0f}
├── Performa 1 Minggu: {pct(current, float(price_1w)):+.2f}%
├── Performa 1 Bulan : {pct(current, float(price_1m)):+.2f}%
└── Performa 3 Bulan : {pct(current, float(price_3m)):+.2f}%

📈 INDIKATOR TEKNIKAL
├── RSI (14)         : {latest['RSI']:.2f}
├── MACD             : {latest['MACD']:.4f} | Signal: {latest['MACD_Signal']:.4f}
├── MACD Histogram   : {latest['MACD_Hist']:.4f}
├── Stochastic K/D   : {latest['Stoch_K']:.2f} / {latest['Stoch_D']:.2f}
├── Williams %R      : {latest['Williams_R']:.2f}
├── CCI              : {latest['CCI']:.2f}
├── ROC (12)         : {latest['ROC']:.2f}%
├── SMA 20           : Rp {latest['SMA_20']:,.0f}
├── SMA 50           : Rp {latest['SMA_50']:,.0f}
├── EMA 12           : Rp {latest['EMA_12']:,.0f}
├── EMA 26           : Rp {latest['EMA_26']:,.0f}
├── Bollinger Upper  : Rp {latest['BB_Upper']:,.0f}
├── Bollinger Mid    : Rp {latest['BB_Mid']:,.0f}
├── Bollinger Lower  : Rp {latest['BB_Lower']:,.0f}
├── BB Position      : {latest['BB_Position']:.3f} (0=lower, 1=upper)
├── ATR (14)         : {latest['ATR']:.2f}
└── Volume Ratio     : {latest['Volume_Ratio']:.2f}x (vs MA20)

🏢 DATA FUNDAMENTAL
├── Market Cap       : {fmt(fundamental.get('Market Cap'))}
├── P/E Ratio        : {fmt(fundamental.get('P/E Ratio'))}
├── P/B Ratio        : {fmt(fundamental.get('P/B Ratio'))}
├── EPS              : {fmt(fundamental.get('EPS'))}
├── Dividend Yield   : {fmt(fundamental.get('Dividend Yield'), is_pct=True)}
├── ROE              : {fmt(fundamental.get('ROE'), is_pct=True)}
├── ROA              : {fmt(fundamental.get('ROA'), is_pct=True)}
├── Profit Margin    : {fmt(fundamental.get('Profit Margin'), is_pct=True)}
├── Revenue Growth   : {fmt(fundamental.get('Revenue Growth'), is_pct=True)}
├── Earnings Growth  : {fmt(fundamental.get('Earnings Growth'), is_pct=True)}
├── Debt/Equity      : {fmt(fundamental.get('Debt to Equity'))}
├── Current Ratio    : {fmt(fundamental.get('Current Ratio'))}
├── Beta             : {fmt(fundamental.get('Beta'))}
├── 52W High         : {fmt(fundamental.get('52W High'))}
└── 52W Low          : {fmt(fundamental.get('52W Low'))}

📰 BERITA TERKINI SAHAM
{news_text}

🌐 BERITA MAKRO EKONOMI INDONESIA
{market_news}

═══════════════════════════════════════════════════════
INSTRUKSI: Berikan analisis LENGKAP dengan format berikut:
═══════════════════════════════════════════════════════

## 1. RINGKASAN KONDISI SAHAM
[Deskripsikan kondisi umum saham saat ini]

## 2. ANALISIS TEKNIKAL
### Tren Utama
### Momentum  
### Volatilitas & Volume
### Support & Resistance
- Support 1: Rp ...
- Support 2: Rp ...
- Resistance 1: Rp ...
- Resistance 2: Rp ...

## 3. ANALISIS FUNDAMENTAL
[Evaluasi valuasi dan kesehatan keuangan]

## 4. ANALISIS SENTIMEN BERITA
[Dampak berita terhadap saham]

## 5. PREDIKSI PERGERAKAN HARGA

### 📅 7 HARI KE DEPAN
- **Prediksi Arah**: [NAIK/TURUN/SIDEWAYS]
- **Target Harga**: Rp ... - Rp ...
- **Probabilitas**: ...%
- **Alasan**: [3-5 poin]

### 📅 30 HARI KE DEPAN
- **Prediksi Arah**: [NAIK/TURUN/SIDEWAYS]
- **Target Harga**: Rp ... - Rp ...
- **Probabilitas**: ...%
- **Alasan**: [3-5 poin]

## 6. REKOMENDASI

### 🎯 REKOMENDASI: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]

| Parameter | Detail |
|-----------|--------|
| Entry Price | Rp ... |
| Target 7 Hari | Rp ... |
| Target 30 Hari | Rp ... |
| Stop Loss | Rp ... |
| Risk/Reward | ... |
| Confidence | ...% |

### Strategi per Profil Investor:
- **Trader Pendek (< 1 minggu)**: ...
- **Swing Trader (1-4 minggu)**: ...
- **Investor Panjang (> 3 bulan)**: ...

## 7. RISIKO UTAMA
[3-5 risiko yang perlu diwaspadai]

## 8. KESIMPULAN
[Paragraf penutup]

---
⚠️ DISCLAIMER: Analisis ini hanya untuk edukasi, bukan saran investasi resmi.
"""


def analyze_with_gemini(model, ticker, stock_name, current_price,
                        df, fundamental, news_text, market_news) -> str:
    prompt = build_prompt(ticker, stock_name, current_price,
                          df, fundamental, news_text, market_news)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )
        return response.text
    except Exception as e:
        return f"❌ Error Gemini API: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Konfigurasi")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Masukkan Gemini API Key Anda",
    )

    st.divider()
    st.subheader("📊 Pilih Saham")
    stocks = get_indonesian_stocks()

    selected_stock_name = st.selectbox(
        "Saham Populer", options=list(stocks.keys()), index=0
    )

    custom_ticker = st.text_input(
        "Atau kode manual (tanpa .JK)",
        placeholder="Contoh: ACES, SIDO",
    )

    if custom_ticker:
        ticker = f"{custom_ticker.upper().strip()}.JK"
        stock_display_name = custom_ticker.upper().strip()
    else:
        ticker = stocks[selected_stock_name]
        stock_display_name = selected_stock_name

    st.divider()
    st.subheader("📅 Periode Data")
    period = st.select_slider(
        "Periode Historis",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )

    st.divider()
    st.subheader("🔧 Opsi Analisis")
    include_news = st.checkbox("📰 Sertakan Berita", value=True)
    include_macro = st.checkbox("🌐 Berita Makro", value=True)
    include_fundamental = st.checkbox("🏢 Data Fundamental", value=True)

    st.divider()
    analyze_btn = st.button("🚀 Analisis Sekarang", type="primary",
                            use_container_width=True)
    st.divider()
    st.caption("⚠️ Bukan saran investasi resmi")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDASI API KEY
# ══════════════════════════════════════════════════════════════════════════════

if not api_key:
    st.warning("⚠️ Masukkan Gemini API Key di sidebar.")
    st.info("""
    **Cara mendapatkan API Key:**
    1. Kunjungi [Google AI Studio](https://aistudio.google.com/)
    2. Login dengan akun Google
    3. Klik **"Get API Key"**
    4. Copy & paste di sidebar kiri

    Atau set di Streamlit Secrets:
    ```
    GEMINI_API_KEY = "your_key_here"
    ```
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA UTAMA
# ══════════════════════════════════════════════════════════════════════════════

try:
    with st.spinner(f"⏳ Memuat data {ticker}..."):
        df = fetch_stock_data(ticker, period)
        if df is None or df.empty:
            st.error(f"❌ Data tidak ditemukan untuk **{ticker}**. "
                     "Pastikan kode saham benar.")
            st.stop()
        df = calculate_all_indicators(df)
        current_price = get_current_price(ticker)
        indicator_summary = get_indicator_summary(df)
except Exception as e:
    st.error(f"❌ Error memuat data: {str(e)}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HEADER
# ══════════════════════════════════════════════════════════════════════════════

price = current_price["current_price"]
change = current_price["change"]
change_pct = current_price["change_pct"]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_color = "normal" if change >= 0 else "inverse"
    st.metric("💰 Harga", f"Rp {price:,.0f}",
              f"{change:+.0f} ({change_pct:+.2f}%)", delta_color=delta_color)

with col2:
    rsi_val = df["RSI"].iloc[-1]
    rsi_label = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Normal"
    st.metric("📊 RSI (14)", f"{rsi_val:.1f}", rsi_label)

with col3:
    macd_val = df["MACD"].iloc[-1]
    macd_sig = df["MACD_Signal"].iloc[-1]
    st.metric("📈 MACD", f"{macd_val:.4f}",
              "🚀 Bullish" if macd_val > macd_sig else "⚠️ Bearish")

with col4:
    vol_ratio = df["Volume_Ratio"].iloc[-1]
    st.metric("📦 Volume", f"{vol_ratio:.2f}x",
              "🔥 Tinggi" if vol_ratio > 1.5 else "📉 Rendah" if vol_ratio < 0.5 else "Normal")

with col5:
    overall = indicator_summary["_summary"]["overall"]
    bull = indicator_summary["_summary"]["bullish_count"]
    bear = indicator_summary["_summary"]["bearish_count"]
    emoji = "🟢" if overall == "BULLISH" else "🔴" if overall == "BEARISH" else "🟡"
    st.metric("🎯 Sinyal", f"{emoji} {overall}", f"Bull:{bull} Bear:{bear}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Chart & Teknikal",
    "🏢 Fundamental",
    "📰 Berita",
    "🤖 Analisis AI",
    "📋 Sinyal",
])


# ─── TAB 1: CHART ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"📈 Chart {stock_display_name}")

    c1, c2 = st.columns([3, 1])
    with c2:
        chart_type = st.radio("Tipe", ["Candlestick", "Line"], horizontal=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_ma = st.checkbox("Moving Averages", value=True)
        show_vol = st.checkbox("Volume", value=True)

    rows = 3 if show_vol else 2
    row_heights = [0.55, 0.25, 0.20] if show_vol else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=(["Harga", "MACD", "Volume"] if show_vol
                        else ["Harga", "MACD"]),
    )

    # Price
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(color="#00d4ff", width=2)
        ), row=1, col=1)

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA20",
            line=dict(color="#ffaa00", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA50",
            line=dict(color="#ff6b6b", width=1.5, dash="dot")), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(100,200,255,0.5)", width=1),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(100,200,255,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(100,200,255,0.05)",
            showlegend=False), row=1, col=1)

    # MACD
    colors_macd = ["#00ff88" if v >= 0 else "#ff4444"
                   for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
        marker_color=colors_macd, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#00d4ff", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#ff6b6b", width=1.5)), row=2, col=1)

    # Volume
    if show_vol:
        vol_colors = ["#00ff88" if c >= o else "#ff4444"
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA20"],
            name="Vol MA20", line=dict(color="#ffaa00", width=1.5)), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI & Stochastic
    st.subheader("📊 RSI & Stochastic")
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=["RSI (14)", "Stochastic (14,3)"])
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#00d4ff", width=2)), row=1, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=1, col=1)
    fig2.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K",
        line=dict(color="#00ff88", width=2)), row=1, col=2)
    fig2.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D",
        line=dict(color="#ffaa00", width=1.5, dash="dot")), row=1, col=2)
    fig2.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=2)
    fig2.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=2)
    fig2.update_layout(template="plotly_dark", height=300,
                       margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 2: FUNDAMENTAL ───────────────────────────────────────────────────────
with tab2:
    st.subheader("🏢 Data Fundamental")
    if include_fundamental:
        with st.spinner("Memuat fundamental..."):
            fundamental = fetch_stock_info(ticker)

        def show_val(v, is_pct=False):
            if v in ("N/A", None):
                return "N/A"
            try:
                fv = float(v)
                if is_pct:
                    return f"{fv*100:.2f}%"
                if fv > 1_000_000_000_000:
                    return f"Rp {fv/1_000_000_000_000:.2f}T"
                return f"{fv:.2f}"
            except Exception:
                return str(v)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### 📊 Valuasi")
            for k in ["P/E Ratio", "P/B Ratio", "EPS", "Market Cap", "Beta"]:
                a, b = st.columns(2)
                a.write(f"**{k}**")
                b.write(show_val(fundamental.get(k)))

            st.markdown("#### 📈 Pertumbuhan")
            for k in ["Revenue Growth", "Earnings Growth",
                      "Profit Margin", "Operating Margin"]:
                a, b = st.columns(2)
                a.write(f"**{k}**")
                b.write(show_val(fundamental.get(k), is_pct=True))

        with col_f2:
            st.markdown("#### 💪 Kesehatan Keuangan")
            for k, pct_flag in [("ROE", True), ("ROA", True),
                                 ("Debt to Equity", False),
                                 ("Current Ratio", False),
                                 ("Dividend Yield", True)]:
                a, b = st.columns(2)
                a.write(f"**{k}**")
                b.write(show_val(fundamental.get(k), is_pct=pct_flag))

            st.markdown("#### 📉 Range Harga")
            for k in ["52W High", "52W Low", "Avg Volume"]:
                a, b = st.columns(2)
                a.write(f"**{k}**")
                b.write(show_val(fundamental.get(k)))
    else:
        st.info("Aktifkan 'Data Fundamental' di sidebar.")


# ─── TAB 3: BERITA ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📰 Berita Terkini")
    if include_news:
        with st.spinner("Memuat berita..."):
            news_articles = fetch_stock_news(stock_display_name, ticker)
            market_articles = fetch_market_news() if include_macro else []

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown(f"#### 📌 Berita {stock_display_name}")
            if news_articles:
                for art in news_articles:
                    title = art["title"]
                    label = (f"{title[:75]}..." if len(title) > 75 else title)
                    with st.expander(f"{'🔴' if art.get('relevant') else '⚪'} {label}"):
                        st.write(f"**Sumber:** {art['source']}")
                        st.write(f"**Tanggal:** {art['published']}")
                        st.write(art["summary"])
                        if art["link"]:
                            st.markdown(f"[🔗 Baca Selengkapnya]({art['link']})")
            else:
                st.info("Tidak ada berita spesifik ditemukan.")

        with col_n2:
            st.markdown("#### 🌐 Berita Makro")
            if market_articles:
                for art in market_articles:
                    title = art["title"]
                    label = (f"{title[:75]}..." if len(title) > 75 else title)
                    with st.expander(label):
                        st.write(f"**Sumber:** {art['source']}")
                        st.write(f"**Tanggal:** {art['published']}")
                        st.write(art["summary"])
                        if art["link"]:
                            st.markdown(f"[🔗 Baca Selengkapnya]({art['link']})")
            else:
                st.info("Aktifkan 'Berita Makro' di sidebar.")
    else:
        st.info("Aktifkan 'Sertakan Berita' di sidebar.")


# ─── TAB 4: ANALISIS AI ───────────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Analisis AI - Gemini 2.5 Flash")

    if "ai_analysis" not in st.session_state:
        st.session_state.ai_analysis = None
        st.session_state.analyzed_ticker = None

    if analyze_btn:
        with st.spinner("🤖 Gemini sedang menganalisis... (30-60 detik)"):
            try:
                model = setup_gemini(api_key)

                fund_data = fetch_stock_info(ticker) if include_fundamental else {}
                stock_news = fetch_stock_news(stock_display_name, ticker) if include_news else []
                macro_news = fetch_market_news() if include_macro else []

                analysis = analyze_with_gemini(
                    model=model,
                    ticker=ticker,
                    stock_name=stock_display_name,
                    current_price=current_price,
                    df=df,
                    fundamental=fund_data,
                    news_text=format_news_for_llm(stock_news),
                    market_news=format_news_for_llm(macro_news),
                )
                st.session_state.ai_analysis = analysis
                st.session_state.analyzed_ticker = ticker

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if st.session_state.ai_analysis:
        st.success(f"✅ Analisis selesai untuk **{stock_display_name}**")
        st.markdown(st.session_state.ai_analysis)
        st.download_button(
            "📥 Download Analisis (TXT)",
            data=st.session_state.ai_analysis,
            file_name=f"analisis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
    else:
        st.info("""
        👆 Klik **"🚀 Analisis Sekarang"** di sidebar untuk memulai.

        **Yang dianalisis:**
        - ✅ Harga real-time & performa historis
        - ✅ 10+ indikator teknikal
        - ✅ Data fundamental perusahaan  
        - ✅ Berita terkini saham
        - ✅ Kondisi makro ekonomi Indonesia

        **Output:**
        - 🎯 Prediksi 7 & 30 hari ke depan
        - 💡 Rekomendasi Buy/Hold/Sell
        - 📊 Level Support & Resistance
        - ⚠️ Analisis risiko lengkap
        """)


# ─── TAB 5: SINYAL ────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Ringkasan Sinyal Indikator")

    summary = indicator_summary["_summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Bullish", summary["bullish_count"])
    c2.metric("🔴 Bearish", summary["bearish_count"])
    overall_emoji = ("🟢" if summary["overall"] == "BULLISH"
                     else "🔴" if summary["overall"] == "BEARISH" else "🟡")
    c3.metric("🎯 Overall", f"{overall_emoji} {summary['overall']}")

    st.divider()

    signal_rows = []
    for name, data in indicator_summary.items():
        if name == "_summary":
            continue
        bias_emoji = ("🟢" if data["bias"] == "BULLISH"
                      else "🔴" if data["bias"] == "BEARISH" else "🟡")
        signal_rows.append({
            "Indikator": name,
            "Nilai": data["value"],
            "Sinyal": data["signal"],
            "Bias": f"{bias_emoji} {data['bias']}",
        })

    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True,
                 hide_index=True)

    st.divider()
    st.subheader("📊 Data Historis (20 Hari Terakhir)")
    cols_show = ["Open", "High", "Low", "Close", "Volume",
                 "RSI", "MACD", "SMA_20", "BB_Upper", "BB_Lower"]
    hist_df = df[cols_show].tail(20).round(2).copy()
    hist_df.index = hist_df.index.strftime("%Y-%m-%d")
    st.dataframe(hist_df, use_container_width=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.8rem;'>
⚠️ <b>DISCLAIMER:</b> Hanya untuk edukasi. Bukan saran investasi resmi.<br>
📈 Powered by Gemini 2.5 Flash • yfinance • Streamlit
</div>
""", unsafe_allow_html=True)
