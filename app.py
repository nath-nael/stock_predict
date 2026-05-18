import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import time

from modules.stock_data import (
    get_indonesian_stocks,
    fetch_stock_data,
    fetch_stock_info,
    get_current_price,
)
from modules.technical_indicators import calculate_all_indicators, get_indicator_summary
from modules.news_scraper import (
    fetch_rss_news,
    fetch_google_news,
    get_market_sentiment_news,
    format_news_for_llm,
)
from modules.llm_analyzer import setup_gemini, analyze_stock

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇩 Analisis Saham Indonesia AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
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
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #00d4ff;
    }
    .signal-bullish { color: #00ff88; font-weight: bold; }
    .signal-bearish { color: #ff4444; font-weight: bold; }
    .signal-neutral { color: #ffaa00; font-weight: bold; }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🇮🇩 Analisis Saham Indonesia</h1>
    <p>Powered by Gemini 2.5 Flash AI • Real-time Technical & Fundamental Analysis</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    # API Key
    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Masukkan Gemini API Key Anda",
    )

    st.divider()

    # Stock Selection
    st.subheader("📊 Pilih Saham")
    stocks = get_indonesian_stocks()
    stock_options = list(stocks.keys())

    selected_stock_name = st.selectbox(
        "Saham",
        options=stock_options,
        index=0,
    )

    # Custom ticker
    custom_ticker = st.text_input(
        "Atau masukkan kode saham manual",
        placeholder="Contoh: ACES, SIDO, MAPI",
        help="Masukkan kode saham tanpa .JK",
    )

    if custom_ticker:
        ticker = f"{custom_ticker.upper()}.JK"
        stock_display_name = custom_ticker.upper()
    else:
        ticker = stocks[selected_stock_name]
        stock_display_name = selected_stock_name

    st.divider()

    # Period Selection
    st.subheader("📅 Periode Data")
    period = st.select_slider(
        "Periode Historis",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )

    st.divider()

    # Analysis Options
    st.subheader("🔧 Opsi Analisis")
    include_news = st.checkbox("📰 Sertakan Berita", value=True)
    include_macro = st.checkbox("🌐 Sertakan Berita Makro", value=True)
    include_fundamental = st.checkbox("🏢 Sertakan Data Fundamental", value=True)

    st.divider()

    # Analyze Button
    analyze_btn = st.button(
        "🚀 Analisis Sekarang",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption("⚠️ Disclaimer: Bukan saran investasi resmi")


# ─── Main Content ─────────────────────────────────────────────────────────────
if not api_key:
    st.warning("⚠️ Masukkan Gemini API Key di sidebar untuk memulai analisis.")
    st.info("""
    **Cara mendapatkan API Key:**
    1. Kunjungi [Google AI Studio](https://aistudio.google.com/)
    2. Login dengan akun Google
    3. Klik "Get API Key"
    4. Copy dan paste di sidebar
    """)
    st.stop()


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache 5 menit
def load_stock_data(ticker, period):
    df = fetch_stock_data(ticker, period)
    df = calculate_all_indicators(df)
    return df


@st.cache_data(ttl=300)
def load_fundamental(ticker):
    return fetch_stock_info(ticker)


@st.cache_data(ttl=600)  # Cache 10 menit
def load_news(stock_name, ticker):
    clean_name = stock_name.split(" - ")[0] if " - " in stock_name else stock_name
    rss_news = fetch_rss_news(clean_name, ticker, max_articles=8)
    google_news = fetch_google_news(f"{clean_name} saham", max_articles=5)
    all_news = rss_news + google_news
    return all_news


@st.cache_data(ttl=600)
def load_market_news():
    return get_market_sentiment_news()


# ─── Auto Load Data ───────────────────────────────────────────────────────────
try:
    with st.spinner(f"⏳ Memuat data {ticker}..."):
        df = load_stock_data(ticker, period)
        current_price = get_current_price(ticker)
        indicator_summary = get_indicator_summary(df)

    if df.empty:
        st.error(f"❌ Tidak dapat memuat data untuk {ticker}. Pastikan kode saham benar.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()


# ─── Price Header ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

price = current_price["current_price"]
change = current_price["change"]
change_pct = current_price["change_pct"]
color = "normal" if change >= 0 else "inverse"

with col1:
    st.metric("💰 Harga Saat Ini", f"Rp {price:,.0f}",
              f"{change:+.0f} ({change_pct:+.2f}%)", delta_color=color)

with col2:
    rsi_val = df["RSI"].iloc[-1]
    rsi_status = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Normal"
    st.metric("📊 RSI (14)", f"{rsi_val:.1f}", rsi_status)

with col3:
    macd_val = df["MACD"].iloc[-1]
    macd_signal = df["MACD_Signal"].iloc[-1]
    macd_status = "🚀 Bullish" if macd_val > macd_signal else "⚠️ Bearish"
    st.metric("📈 MACD", f"{macd_val:.4f}", macd_status)

with col4:
    vol_ratio = df["Volume_Ratio"].iloc[-1]
    vol_status = "🔥 Tinggi" if vol_ratio > 1.5 else "📉 Rendah" if vol_ratio < 0.5 else "Normal"
    st.metric("📦 Volume Ratio", f"{vol_ratio:.2f}x", vol_status)

with col5:
    overall = indicator_summary["_summary"]["overall"]
    bull = indicator_summary["_summary"]["bullish_count"]
    bear = indicator_summary["_summary"]["bearish_count"]
    overall_emoji = "🟢" if overall == "BULLISH" else "🔴" if overall == "BEARISH" else "🟡"
    st.metric("🎯 Sinyal Overall", f"{overall_emoji} {overall}", f"Bull:{bull} Bear:{bear}")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Chart & Teknikal",
    "🏢 Fundamental",
    "📰 Berita",
    "🤖 Analisis AI",
    "📋 Sinyal Indikator",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHART & TEKNIKAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"📈 Chart {stock_display_name}")

    # Chart Type Selection
    chart_col1, chart_col2 = st.columns([3, 1])
    with chart_col2:
        chart_type = st.radio("Tipe Chart", ["Candlestick", "Line"], horizontal=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_ma = st.checkbox("Moving Averages", value=True)
        show_volume = st.checkbox("Volume", value=True)

    # Build Chart
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]
    subplot_titles = ["Harga", "MACD", "Volume & RSI"] if show_volume else ["Harga", "MACD"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Price Chart ──
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name="Close", line=dict(color="#00d4ff", width=2)
        ), row=1, col=1)

    # ── Moving Averages ──
    if show_ma:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"],
            name="SMA 20", line=dict(color="#ffaa00", width=1.5, dash="dot")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"],
            name="SMA 50", line=dict(color="#ff6b6b", width=1.5, dash="dot")
        ), row=1, col=1)

    # ── Bollinger Bands ──
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            name="BB Upper", line=dict(color="rgba(100,200,255,0.5)", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            name="BB Lower", line=dict(color="rgba(100,200,255,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(100,200,255,0.05)",
            showlegend=False,
        ), row=1, col=1)

    # ── MACD ──
    colors_macd = ["#00ff88" if v >= 0 else "#ff4444" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        name="MACD Hist", marker_color=colors_macd, opacity=0.7
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#00d4ff", width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal", line=dict(color="#ff6b6b", width=1.5)
    ), row=2, col=1)

    # ── Volume ──
    if show_volume:
        vol_colors = ["#00ff88" if c >= o else "#ff4444"
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color=vol_colors, opacity=0.7
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_MA20"],
            name="Vol MA20", line=dict(color="#ffaa00", width=1.5)
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    st.plotly_chart(fig, use_container_width=True)

    # ── RSI Chart ──
    st.subheader("📊 RSI & Stochastic")
    fig_rsi = make_subplots(rows=1, cols=2, subplot_titles=["RSI (14)", "Stochastic (14,3)"])

    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#00d4ff", width=2)
    ), row=1, col=1)
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=1, col=1)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=1, col=1)

    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df["Stoch_K"], name="Stoch %K",
        line=dict(color="#00ff88", width=2)
    ), row=1, col=2)
    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df["Stoch_D"], name="Stoch %D",
        line=dict(color="#ffaa00", width=1.5, dash="dot")
    ), row=1, col=2)
    fig_rsi.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=2)
    fig_rsi.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=2)

    fig_rsi.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_rsi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FUNDAMENTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏢 Data Fundamental")

    if include_fundamental:
        with st.spinner("Memuat data fundamental..."):
            fundamental = load_fundamental(ticker)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("#### 📊 Valuasi")
            val_data = {
                "P/E Ratio": fundamental.get("P/E Ratio", "N/A"),
                "P/B Ratio": fundamental.get("P/B Ratio", "N/A"),
                "EPS": fundamental.get("EPS", "N/A"),
                "Market Cap": fundamental.get("Market Cap", "N/A"),
                "Beta": fundamental.get("Beta", "N/A"),
            }
            for k, v in val_data.items():
                col_a, col_b = st.columns([1, 1])
                col_a.write(f"**{k}**")
                col_b.write(str(v))

            st.markdown("#### 📈 Pertumbuhan")
            growth_data = {
                "Revenue Growth": fundamental.get("Revenue Growth", "N/A"),
                "Earnings Growth": fundamental.get("Earnings Growth", "N/A"),
                "Profit Margin": fundamental.get("Profit Margin", "N/A"),
                "Operating Margin": fundamental.get("Operating Margin", "N/A"),
            }
            for k, v in growth_data.items():
                col_a, col_b = st.columns([1, 1])
                col_a.write(f"**{k}**")
                try:
                    if v != "N/A" and v is not None:
                        col_b.write(f"{float(v)*100:.2f}%")
                    else:
                        col_b.write("N/A")
                except:
                    col_b.write(str(v))

        with col_f2:
            st.markdown("#### 💪 Kesehatan Keuangan")
            health_data = {
                "ROE": fundamental.get("ROE", "N/A"),
                "ROA": fundamental.get("ROA", "N/A"),
                "Debt/Equity": fundamental.get("Debt to Equity", "N/A"),
                "Current Ratio": fundamental.get("Current Ratio", "N/A"),
                "Dividend Yield": fundamental.get("Dividend Yield", "N/A"),
            }
            for k, v in health_data.items():
                col_a, col_b = st.columns([1, 1])
                col_a.write(f"**{k}**")
                try:
                    if v != "N/A" and v is not None and k in ["ROE", "ROA", "Dividend Yield"]:
                        col_b.write(f"{float(v)*100:.2f}%")
                    else:
                        col_b.write(str(v) if v else "N/A")
                except:
                    col_b.write(str(v))

            st.markdown("#### 📉 Range Harga")
            range_data = {
                "52W High": fundamental.get("52W High", "N/A"),
                "52W Low": fundamental.get("52W Low", "N/A"),
                "Avg Volume": fundamental.get("Avg Volume", "N/A"),
            }
            for k, v in range_data.items():
                col_a, col_b = st.columns([1, 1])
                col_a.write(f"**{k}**")
                col_b.write(str(v))
    else:
        st.info("Data fundamental dinonaktifkan. Aktifkan di sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: BERITA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📰 Berita Terkini")

    if include_news:
        with st.spinner("Memuat berita..."):
            news_articles = load_news(stock_display_name, ticker)
            market_news_articles = load_market_news() if include_macro else []

        col_n1, col_n2 = st.columns([1, 1])

        with col_n1:
            st.markdown(f"#### 📌 Berita {stock_display_name}")
            if news_articles:
                for article in news_articles[:8]:
                    with st.expander(
                        f"{'🔴' if article.get('relevant') else '⚪'} {article['title'][:80]}..."
                        if len(article['title']) > 80 else article['title']
                    ):
                        st.write(f"**Sumber:** {article['source']}")
                        st.write(f"**Tanggal:** {article['published']}")
                        st.write(article['summary'])
                        if article['link']:
                            st.markdown(f"[🔗 Baca Selengkapnya]({article['link']})")
            else:
                st.info("Tidak ada berita spesifik yang ditemukan.")

        with col_n2:
            st.markdown("#### 🌐 Berita Makro Ekonomi")
            if market_news_articles:
                for article in market_news_articles[:8]:
                    with st.expander(
                        f"{article['title'][:80]}..."
                        if len(article['title']) > 80 else article['title']
                    ):
                        st.write(f"**Sumber:** {article['source']}")
                        st.write(f"**Tanggal:** {article['published']}")
                        st.write(article['summary'])
                        if article['link']:
                            st.markdown(f"[🔗 Baca Selengkapnya]({article['link']})")
            else:
                st.info("Tidak ada berita makro yang ditemukan.")
    else:
        st.info("Berita dinonaktifkan. Aktifkan di sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANALISIS AI
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Analisis AI - Gemini 2.5 Flash")

    if "ai_analysis" not in st.session_state:
        st.session_state.ai_analysis = None
        st.session_state.analyzed_ticker = None

    if analyze_btn or (st.session_state.analyzed_ticker != ticker):
        if analyze_btn:
            st.session_state.ai_analysis = None

    if analyze_btn:
        with st.spinner("🤖 Gemini sedang menganalisis... (mungkin 30-60 detik)"):
            try:
                # Setup Gemini
                model = setup_gemini(api_key)

                # Load semua data
                fundamental_data = load_fundamental(ticker) if include_fundamental else {}

                news_articles = load_news(stock_display_name, ticker) if include_news else []
                market_news_articles = load_market_news() if include_macro else []

                news_text = format_news_for_llm(news_articles)
                market_news_text = format_news_for_llm(market_news_articles)

                indicator_sum = get_indicator_summary(df)

                # Analisis dengan Gemini
                analysis = analyze_stock(
                    model=model,
                    ticker=ticker,
                    stock_name=stock_display_name,
                    current_price=current_price,
                    df=df,
                    indicators=indicator_sum,
                    fundamental=fundamental_data,
                    news_text=news_text,
                    market_news=market_news_text,
                )

                st.session_state.ai_analysis = analysis
                st.session_state.analyzed_ticker = ticker

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if st.session_state.ai_analysis:
        # Display Analysis
        st.success(f"✅ Analisis selesai untuk {stock_display_name}")

        # Render markdown analysis
        st.markdown(st.session_state.ai_analysis)

        # Download button
        st.download_button(
            label="📥 Download Analisis (TXT)",
            data=st.session_state.ai_analysis,
            file_name=f"analisis_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
    else:
        st.info("""
        👆 Klik **"🚀 Analisis Sekarang"** di sidebar untuk memulai analisis AI.

        **Yang akan dianalisis:**
        - ✅ Data harga real-time
        - ✅ 10+ indikator teknikal
        - ✅ Data fundamental perusahaan
        - ✅ Berita terkini saham
        - ✅ Kondisi makro ekonomi Indonesia

        **Output:**
        - 🎯 Prediksi 7 hari ke depan
        - 🎯 Prediksi 30 hari ke depan
        - 💡 Rekomendasi Buy/Hold/Sell
        - 📊 Level Support & Resistance
        - ⚠️ Analisis risiko
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: SINYAL INDIKATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Ringkasan Sinyal Indikator")

    summary = indicator_summary["_summary"]

    # Overall Signal
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("🟢 Sinyal Bullish", summary["bullish_count"])
    with col_s2:
        st.metric("🔴 Sinyal Bearish", summary["bearish_count"])
    with col_s3:
        overall_color = "🟢" if summary["overall"] == "BULLISH" else "🔴" if summary["overall"] == "BEARISH" else "🟡"
        st.metric("🎯 Kesimpulan", f"{overall_color} {summary['overall']}")

    st.divider()

    # Individual Signals
    signal_data = []
    for name, data in indicator_summary.items():
        if name == "_summary":
            continue
        bias_emoji = "🟢" if data["bias"] == "BULLISH" else "🔴" if data["bias"] == "BEARISH" else "🟡"
        signal_data.append({
            "Indikator": name,
            "Nilai": data["value"],
            "Sinyal": data["signal"],
            "Bias": f"{bias_emoji} {data['bias']}",
        })

    signal_df = pd.DataFrame(signal_data)
    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Historical Data Table
    st.subheader("📊 Data Historis (20 Hari Terakhir)")
    display_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA_20", "BB_Upper", "BB_Lower"]
    display_df = df[display_cols].tail(20).round(2)
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    ⚠️ <b>DISCLAIMER:</b> Aplikasi ini hanya untuk tujuan edukasi dan penelitian. 
    Bukan merupakan saran investasi resmi. Selalu lakukan riset mandiri 
    dan konsultasikan dengan financial advisor sebelum berinvestasi.<br><br>
    📈 Powered by Gemini 2.5 Flash • yfinance • Streamlit
</div>
""", unsafe_allow_html=True)
