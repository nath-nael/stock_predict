import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re


RSS_FEEDS = {
    "Kontan": "https://rss.kontan.co.id/category/investasi",
    "CNBC Indonesia": "https://www.cnbcindonesia.com/rss",
    "Bisnis.com": "https://rss.bisnis.com/feed/rss2/finansial/market",
    "IDX Channel": "https://www.idxchannel.com/feed",
    "Investing.com ID": "https://id.investing.com/rss/news.rss",
}


def fetch_rss_news(stock_name: str, ticker: str, max_articles: int = 10) -> list:
    """Ambil berita dari RSS feeds"""
    articles = []

    # Bersihkan ticker untuk pencarian
    clean_ticker = ticker.replace(".JK", "")
    search_terms = [clean_ticker, stock_name.split(" - ")[0]]

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                published = entry.get("published", "")

                # Filter relevan dengan saham
                combined_text = (title + " " + summary).upper()
                is_relevant = any(
                    term.upper() in combined_text for term in search_terms
                )

                if is_relevant or len(articles) < 3:  # Ambil minimal 3 berita umum
                    articles.append({
                        "source": source,
                        "title": title,
                        "summary": clean_html(summary)[:300],
                        "link": link,
                        "published": published,
                        "relevant": is_relevant,
                    })

                if len(articles) >= max_articles:
                    break

        except Exception as e:
            continue

    # Sort: relevan dulu
    articles.sort(key=lambda x: x["relevant"], reverse=True)
    return articles[:max_articles]


def fetch_google_news(query: str, max_articles: int = 8) -> list:
    """Ambil berita dari Google News RSS"""
    articles = []
    try:
        encoded_query = requests.utils.quote(f"{query} saham Indonesia")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=id&gl=ID&ceid=ID:id"

        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles]:
            articles.append({
                "source": "Google News",
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "relevant": True,
            })
    except Exception as e:
        pass

    return articles


def get_market_sentiment_news() -> list:
    """Ambil berita makro/market umum"""
    articles = []
    macro_queries = ["IHSG", "Bank Indonesia suku bunga", "ekonomi Indonesia"]

    for query in macro_queries:
        news = fetch_google_news(query, max_articles=3)
        articles.extend(news)

    return articles[:10]


def clean_html(text: str) -> str:
    """Bersihkan HTML tags dari teks"""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text()
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def format_news_for_llm(articles: list) -> str:
    """Format berita untuk dikirim ke LLM"""
    if not articles:
        return "Tidak ada berita terkini yang ditemukan."

    formatted = []
    for i, article in enumerate(articles, 1):
        relevant_tag = "[RELEVAN]" if article.get("relevant") else "[UMUM]"
        formatted.append(
            f"{i}. {relevant_tag} {article['source']}\n"
            f"   Judul: {article['title']}\n"
            f"   Ringkasan: {article['summary']}\n"
            f"   Tanggal: {article['published']}\n"
        )

    return "\n".join(formatted)
