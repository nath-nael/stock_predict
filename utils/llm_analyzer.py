import google.generativeai as genai
import pandas as pd
from datetime import datetime


def setup_gemini(api_key: str):
    """Setup Gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def build_analysis_prompt(
    ticker: str,
    stock_name: str,
    current_price: dict,
    df: pd.DataFrame,
    indicators: dict,
    fundamental: dict,
    news_text: str,
    market_news: str,
) -> str:
    """Bangun prompt lengkap untuk analisis LLM"""

    latest = df.iloc[-1]

    # Hitung performa historis
    price_1w = df["Close"].iloc[-5] if len(df) >= 5 else df["Close"].iloc[0]
    price_1m = df["Close"].iloc[-20] if len(df) >= 20 else df["Close"].iloc[0]
    price_3m = df["Close"].iloc[-60] if len(df) >= 60 else df["Close"].iloc[0]

    current = current_price["current_price"]
    perf_1w = ((current - price_1w) / price_1w * 100) if price_1w > 0 else 0
    perf_1m = ((current - price_1m) / price_1m * 100) if price_1m > 0 else 0
    perf_3m = ((current - price_3m) / price_3m * 100) if price_3m > 0 else 0

    # Format fundamental
    def fmt(val, suffix="", is_pct=False):
        if val == "N/A" or val is None:
            return "N/A"
        try:
            if is_pct:
                return f"{float(val)*100:.2f}%"
            if isinstance(val, float) and val > 1_000_000_000:
                return f"Rp {val/1_000_000_000_000:.2f}T"
            return f"{float(val):.2f}{suffix}"
        except:
            return str(val)

    prompt = f"""
Kamu adalah analis saham profesional Indonesia dengan pengalaman 20 tahun di pasar modal IDX.
Analisis komprehensif berikut dan berikan rekomendasi investasi yang actionable.

═══════════════════════════════════════════════════════
📊 DATA SAHAM: {stock_name} ({ticker})
Tanggal Analisis: {datetime.now().strftime("%d %B %Y, %H:%M WIB")}
═══════════════════════════════════════════════════════

💰 HARGA & PERFORMA
├── Harga Saat Ini  : Rp {current:,.0f}
├── Perubahan Hari  : {current_price['change']:+.0f} ({current_price['change_pct']:+.2f}%)
├── Volume Hari Ini : {current_price['volume']:,.0f}
├── Performa 1 Minggu: {perf_1w:+.2f}%
├── Performa 1 Bulan : {perf_1m:+.2f}%
└── Performa 3 Bulan : {perf_3m:+.2f}%

📈 INDIKATOR TEKNIKAL
├── RSI (14)        : {latest['RSI']:.2f} → {'OVERBOUGHT ⚠️' if latest['RSI'] > 70 else 'OVERSOLD 🟢' if latest['RSI'] < 30 else 'NORMAL'}
├── MACD            : {latest['MACD']:.4f} | Signal: {latest['MACD_Signal']:.4f} | Hist: {latest['MACD_Hist']:.4f}
├── Stochastic K/D  : {latest['Stoch_K']:.2f} / {latest['Stoch_D']:.2f}
├── Williams %R     : {latest['Williams_R']:.2f}
├── CCI             : {latest['CCI']:.2f}
├── ROC (12)        : {latest['ROC']:.2f}%
│
├── SMA 20          : Rp {latest['SMA_20']:,.0f}
├── SMA 50          : Rp {latest['SMA_50']:,.0f}
├── EMA 12          : Rp {latest['EMA_12']:,.0f}
├── EMA 26          : Rp {latest['EMA_26']:,.0f}
│
├── Bollinger Upper : Rp {latest['BB_Upper']:,.0f}
├── Bollinger Mid   : Rp {latest['BB_Mid']:,.0f}
├── Bollinger Lower : Rp {latest['BB_Lower']:,.0f}
├── BB Position     : {latest['BB_Position']:.3f} (0=lower, 1=upper)
│
├── ATR (14)        : {latest['ATR']:.2f}
├── Volume          : {latest['Volume']:,.0f}
└── Volume Ratio    : {latest['Volume_Ratio']:.2f}x (vs MA20)

🏢 DATA FUNDAMENTAL
├── Market Cap      : {fmt(fundamental.get('Market Cap'))}
├── P/E Ratio       : {fmt(fundamental.get('P/E Ratio'))}
├── P/B Ratio       : {fmt(fundamental.get('P/B Ratio'))}
├── EPS             : {fmt(fundamental.get('EPS'))}
├── Dividend Yield  : {fmt(fundamental.get('Dividend Yield'), is_pct=True)}
├── ROE             : {fmt(fundamental.get('ROE'), is_pct=True)}
├── ROA             : {fmt(fundamental.get('ROA'), is_pct=True)}
├── Profit Margin   : {fmt(fundamental.get('Profit Margin'), is_pct=True)}
├── Revenue Growth  : {fmt(fundamental.get('Revenue Growth'), is_pct=True)}
├── Earnings Growth : {fmt(fundamental.get('Earnings Growth'), is_pct=True)}
├── Debt/Equity     : {fmt(fundamental.get('Debt to Equity'))}
├── Current Ratio   : {fmt(fundamental.get('Current Ratio'))}
├── Beta            : {fmt(fundamental.get('Beta'))}
├── 52W High        : {fmt(fundamental.get('52W High'))}
└── 52W Low         : {fmt(fundamental.get('52W Low'))}

📰 BERITA TERKINI SAHAM INI
{news_text}

🌐 BERITA MAKRO EKONOMI INDONESIA
{market_news}

═══════════════════════════════════════════════════════
INSTRUKSI ANALISIS:
═══════════════════════════════════════════════════════

Berikan analisis LENGKAP dan TERSTRUKTUR dengan format berikut:

## 1. RINGKASAN KONDISI SAHAM
[Deskripsikan kondisi umum saham saat ini dalam 2-3 kalimat]

## 2. ANALISIS TEKNIKAL
### Tren Utama
[Analisis tren berdasarkan MA, MACD]

### Momentum
[Analisis RSI, Stochastic, Williams %R, CCI]

### Volatilitas & Volume
[Analisis Bollinger Bands, ATR, Volume]

### Support & Resistance Key Levels
- Support 1: Rp ...
- Support 2: Rp ...
- Resistance 1: Rp ...
- Resistance 2: Rp ...

## 3. ANALISIS FUNDAMENTAL
[Evaluasi valuasi, kesehatan keuangan, pertumbuhan]

## 4. ANALISIS SENTIMEN BERITA
[Ringkasan dampak berita terhadap saham]

## 5. PREDIKSI PERGERAKAN HARGA

### 📅 7 HARI KE DEPAN
- **Prediksi Arah**: [NAIK/TURUN/SIDEWAYS]
- **Target Harga**: Rp ... - Rp ...
- **Probabilitas**: ...%
- **Alasan Utama**: [3-5 poin]

### 📅 30 HARI KE DEPAN  
- **Prediksi Arah**: [NAIK/TURUN/SIDEWAYS]
- **Target Harga**: Rp ... - Rp ...
- **Probabilitas**: ...%
- **Alasan Utama**: [3-5 poin]

## 6. REKOMENDASI

### 🎯 REKOMENDASI UTAMA: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]

| Parameter | Detail |
|-----------|--------|
| Entry Price | Rp ... |
| Target Price (7 hari) | Rp ... |
| Target Price (30 hari) | Rp ... |
| Stop Loss | Rp ... |
| Risk/Reward Ratio | ... |
| Confidence Level | ...% |

### Strategi untuk Berbagai Profil Investor:
- **Trader Jangka Pendek (< 1 minggu)**: ...
- **Swing Trader (1-4 minggu)**: ...
- **Investor Jangka Panjang (> 3 bulan)**: ...

## 7. RISIKO UTAMA
[List 3-5 risiko yang perlu diwaspadai]

## 8. KESIMPULAN
[Paragraf penutup yang merangkum semua analisis]

---
⚠️ DISCLAIMER: Analisis ini hanya untuk tujuan edukasi dan bukan merupakan saran investasi resmi. 
Selalu lakukan riset mandiri dan konsultasikan dengan financial advisor sebelum berinvestasi.
"""
    return prompt


def analyze_stock(
    model,
    ticker: str,
    stock_name: str,
    current_price: dict,
    df: pd.DataFrame,
    indicators: dict,
    fundamental: dict,
    news_text: str,
    market_news: str,
) -> str:
    """Kirim data ke Gemini dan dapatkan analisis"""

    prompt = build_analysis_prompt(
        ticker, stock_name, current_price, df,
        indicators, fundamental, news_text, market_news
    )

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
        return f"Error saat menganalisis dengan Gemini: {str(e)}"
