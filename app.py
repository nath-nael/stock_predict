import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Ambil data OHLCV dari yfinance untuk saham Indonesia"""
    # Tambahkan .JK jika belum ada
    if not ticker.endswith(".JK"):
        ticker = ticker + ".JK"

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


def get_stock_info(ticker: str) -> dict:
    """Ambil informasi dasar saham"""
    if not ticker.endswith(".JK"):
        ticker = ticker + ".JK"

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "IDR"),
            "exchange": info.get("exchange", "IDX"),
            "website": info.get("website", "N/A"),
            "description": info.get("longBusinessSummary", "N/A")[:500] if info.get("longBusinessSummary") else "N/A",
        }
    except Exception as e:
        print(f"Error fetching stock info: {e}")
        return {}


def get_current_price(ticker: str) -> dict:
    """Ambil harga terkini"""
    if not ticker.endswith(".JK"):
        ticker = ticker + ".JK"

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2d")

        if hist.empty:
            return {}

        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
        change = current - prev
        change_pct = (change / prev) * 100

        return {
            "current_price": round(current, 2),
            "previous_close": round(prev, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": hist['Volume'].iloc[-1],
            "high_today": round(hist['High'].iloc[-1], 2),
            "low_today": round(hist['Low'].iloc[-1], 2),
        }
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return {}


# Daftar saham populer Indonesia
POPULAR_STOCKS = {
    "BBCA": "Bank Central Asia",
    "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",
    "TLKM": "Telkom Indonesia",
    "ASII": "Astra International",
    "UNVR": "Unilever Indonesia",
    "GOTO": "GoTo Gojek Tokopedia",
    "BYAN": "Bayan Resources",
    "ADRO": "Adaro Energy",
    "ICBP": "Indofood CBP",
    "INDF": "Indofood Sukses Makmur",
    "KLBF": "Kalbe Farma",
    "ANTM": "Aneka Tambang",
    "PTBA": "Bukit Asam",
    "SMGR": "Semen Indonesia",
    "PGAS": "Perusahaan Gas Negara",
    "JSMR": "Jasa Marga",
    "EXCL": "XL Axiata",
    "ISAT": "Indosat Ooredoo",
    "MNCN": "Media Nusantara Citra",
}
