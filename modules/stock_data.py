import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_indonesian_stocks():
    """Daftar saham populer Indonesia"""
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


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Ambil data historis saham dari yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        raise Exception(f"Gagal mengambil data saham: {str(e)}")


def fetch_stock_info(ticker: str) -> dict:
    """Ambil informasi fundamental saham"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamental = {
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

        return fundamental
    except Exception as e:
        return {"Error": str(e)}


def get_current_price(ticker: str) -> dict:
    """Ambil harga terkini"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2d")

        if len(hist) >= 2:
            current = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            change = current - prev
            change_pct = (change / prev) * 100
        else:
            current = info.get("currentPrice", 0)
            change = 0
            change_pct = 0

        return {
            "current_price": round(current, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": hist["Volume"].iloc[-1] if len(hist) > 0 else 0,
        }
    except Exception as e:
        return {"current_price": 0, "change": 0, "change_pct": 0, "volume": 0}
