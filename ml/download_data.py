import yfinance as yf, pandas as pd, os
from datetime import datetime, timedelta
from colorama import Fore, init
init(autoreset=True)

TICKERS = ["NVDA","MSFT","GOOGL","META","AAPL","AMD","TSLA","AMZN",
           "XOM","CVX","LLY","MRNA","PFE","SPY","QQQ","IWM","XLE","XLK"]
END   = datetime.today().strftime("%Y-%m-%d")
START = (datetime.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")
os.makedirs("data/historical", exist_ok=True)

print(f"Downloading clean data for {len(TICKERS)} tickers...")
for ticker in TICKERS:
    try:
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        # Flatten MultiIndex columns yfinance sometimes produces
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw[["open","high","low","close","volume"]]
        raw = raw.dropna()
        raw.to_csv(f"data/historical/{ticker}.csv")
        print(f"  {Fore.GREEN}✓ {ticker}: {len(raw)} rows | close range ${raw.close.min():.0f}-${raw.close.max():.0f}")
    except Exception as e:
        print(f"  {Fore.RED}✗ {ticker}: {e}")
print("Done. Next: python ml/feature_engineering.py")
