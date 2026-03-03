"""
News + Reddit sentiment collector using working free sources.
Run: python ml/sentiment_collector.py
"""
import requests, sqlite3, os, time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from colorama import Fore, init
init(autoreset=True)

os.makedirs("data/sentiment", exist_ok=True)

TICKERS = ["NVDA","MSFT","GOOGL","META","AAPL","AMD","TSLA","AMZN",
           "XOM","CVX","LLY","MRNA","SPY","QQQ"]

analyzer = SentimentIntensityAnalyzer()

conn = sqlite3.connect("data/sentiment/news_sentiment.db")
conn.execute("""CREATE TABLE IF NOT EXISTS headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, headline TEXT, source TEXT,
    compound REAL, positive REAL, negative REAL, neutral REAL,
    timestamp TEXT)""")
conn.commit()

def store(ticker, headline, source, scores):
    conn.execute(
        "INSERT INTO headlines VALUES (NULL,?,?,?,?,?,?,?,?)",
        (ticker, headline, source, scores["compound"],
         scores["pos"], scores["neg"], scores["neu"],
         datetime.now().isoformat()))
    conn.commit()

def label(score):
    if score > 0.05: return f"{Fore.GREEN}BULLISH{Fore.RESET}"
    if score < -0.05: return f"{Fore.RED}BEARISH{Fore.RESET}"
    return f"{Fore.YELLOW}NEUTRAL{Fore.RESET}"

total = 0
headers = {"User-Agent": "Mozilla/5.0 LeapsIntel/1.0"}

# ── Source 1: Reddit via public JSON API (no auth needed) ─────────────────
print(f"{Fore.YELLOW}Scanning Reddit...")
SUBREDDITS = ["wallstreetbets","stocks","investing","options"]
for sub in SUBREDDITS:
    try:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=50"
        r = requests.get(url, headers={"User-Agent":"leaps_intel/1.0"}, timeout=10)
        posts = r.json()["data"]["children"]
        for post in posts:
            title = post["data"]["title"]
            score_val = post["data"]["score"]
            for ticker in TICKERS:
                if f" {ticker} " in f" {title.upper()} " or f"${ticker}" in title.upper():
                    scores = analyzer.polarity_scores(title)
                    store(ticker, title, f"reddit/{sub}", scores)
                    total += 1
                    print(f"  r/{sub} | {ticker} | {label(scores['compound'])} {scores['compound']:+.2f} | {title[:60]}")
        time.sleep(1)
    except Exception as e:
        print(f"  {Fore.YELLOW}Reddit {sub}: {e}")

# ── Source 2: Finviz news scrape ──────────────────────────────────────────
print(f"\n{Fore.YELLOW}Scanning Finviz news...")
for ticker in TICKERS[:8]:  # limit to avoid rate limiting
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        r = requests.get(url, headers=headers, timeout=10)
        from html.parser import HTMLParser
        class NewsParser(HTMLParser):
            def __init__(self):
                super().__init__(); self.headlines=[]; self.in_news=False
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if tag=="a" and "class" in attrs and "tab-link" in str(attrs.get("class","")):
                    self.in_news=True
            def handle_data(self,data):
                if self.in_news and len(data.strip())>20:
                    self.headlines.append(data.strip()); self.in_news=False
        p = NewsParser(); p.feed(r.text)
        for h in p.headlines[:5]:
            scores = analyzer.polarity_scores(h)
            store(ticker, h, "finviz", scores)
            total += 1
            print(f"  Finviz | {ticker} | {label(scores['compound'])} {scores['compound']:+.2f} | {h[:60]}")
        time.sleep(0.5)
    except Exception as e:
        print(f"  {Fore.YELLOW}Finviz {ticker}: {e}")

# ── Source 3: score ticker-specific search terms with VADER directly ──────
print(f"\n{Fore.YELLOW}Generating synthetic historical sentiment from price action...")
import pandas as pd
for ticker in TICKERS:
    try:
        df = pd.read_csv(f"data/historical/{ticker}.csv", index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        # Use price momentum as a proxy for historical sentiment
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)
        df = df.dropna()
        for date, row in df.iterrows():
            # Approximate sentiment from momentum (for historical ML training)
            compound = float(np.clip(row["ret_5d"] * 3, -1, 1))
            conn.execute("INSERT INTO headlines VALUES (NULL,?,?,?,?,?,?,?,?)",
                (ticker, f"price_momentum_{date.date()}", "price_proxy",
                 compound, max(0,compound), max(0,-compound), 1-abs(compound),
                 date.isoformat()))
        conn.commit()
        print(f"  {Fore.GREEN}✓ {ticker}: {len(df)} historical sentiment proxies")
        total += len(df)
    except Exception as e:
        print(f"  {Fore.YELLOW}{ticker}: {e}")

import numpy as np
print(f"\n{Fore.GREEN}✓ Total records stored: {total:,}")
print(f"\n{Fore.YELLOW}Live sentiment by ticker:")
cur = conn.execute("""SELECT ticker, COUNT(*) as n, ROUND(AVG(compound),3) as avg_sent,
    SUM(CASE WHEN compound>0.05 THEN 1 ELSE 0 END) as bull,
    SUM(CASE WHEN compound<-0.05 THEN 1 ELSE 0 END) as bear
    FROM headlines WHERE source != 'price_proxy'
    GROUP BY ticker ORDER BY n DESC""")
rows = cur.fetchall()
if rows:
    for row in rows:
        t,n,avg,bull,bear = row
        print(f"  {t:<6} {n:>3} articles | avg={avg:+.3f} | {Fore.GREEN}{bull}▲{Fore.RESET} {Fore.RED}{bear}▼")
else:
    print(f"  {Fore.YELLOW}No live news captured (Reddit may be throttling). Price proxies stored for ML.")
conn.close()
print(f"\n{Fore.GREEN}✓ Sentiment DB ready at data/sentiment/news_sentiment.db")
