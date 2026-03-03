import pandas as pd, numpy as np, ta, os, warnings
warnings.filterwarnings("ignore")
from colorama import Fore, init
init(autoreset=True)

TICKERS = ["NVDA","MSFT","GOOGL","META","AAPL","AMD","TSLA","AMZN",
           "XOM","CVX","LLY","MRNA","PFE","SPY","QQQ","IWM","XLE","XLK"]
os.makedirs("data/features", exist_ok=True)

def load_csv(ticker):
    df = pd.read_csv(f"data/historical/{ticker}.csv", index_col=0, parse_dates=True)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open","high","low","close","volume"]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    return df

def build_features(df):
    close, high, low, vol, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    df["sma_20"]  = ta.trend.sma_indicator(close, 20)
    df["sma_50"]  = ta.trend.sma_indicator(close, 50)
    df["sma_200"] = ta.trend.sma_indicator(close, 200)
    macd = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["adx"]     = ta.trend.adx(high, low, close)
    df["adx_pos"] = ta.trend.adx_pos(high, low, close)
    df["adx_neg"] = ta.trend.adx_neg(high, low, close)
    df["cci"]     = ta.trend.cci(high, low, close)
    aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
    df["aroon_up"]   = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    df["rsi_14"]  = ta.momentum.rsi(close, 14)
    df["rsi_7"]   = ta.momentum.rsi(close, 7)
    df["stoch_k"] = ta.momentum.stoch(high, low, close)
    df["stoch_d"] = ta.momentum.stoch_signal(high, low, close)
    df["roc_10"]  = ta.momentum.roc(close, 10)
    df["roc_30"]  = ta.momentum.roc(close, 30)
    df["williams"]= ta.momentum.williams_r(high, low, close)
    df["awesome"] = ta.momentum.awesome_oscillator(high, low)
    bb = ta.volatility.BollingerBands(close)
    df["bb_width"]= bb.bollinger_wband()
    df["bb_pct"]  = bb.bollinger_pband()
    df["atr"]     = ta.volatility.average_true_range(high, low, close)
    df["atr_pct"] = df["atr"] / (close + 1e-9)
    df["hv_20"]   = close.pct_change().rolling(20).std() * (252**0.5)
    df["hv_60"]   = close.pct_change().rolling(60).std() * (252**0.5)
    df["hv_90"]   = close.pct_change().rolling(90).std() * (252**0.5)
    df["obv"]     = ta.volume.on_balance_volume(close, vol)
    df["mfi"]     = ta.volume.money_flow_index(high, low, close, vol)
    df["vol_ratio"]= vol / (vol.rolling(20).mean() + 1e-9)
    df["cmf"]     = ta.volume.chaikin_money_flow(high, low, close, vol)
    df["returns_1d"]  = close.pct_change(1)
    df["returns_5d"]  = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)
    df["returns_60d"] = close.pct_change(60)
    df["gap"]         = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
    df["candle_body"] = (close - open_) / (open_ + 1e-9)
    df["above_sma20"]  = (close > df["sma_20"]).astype(int)
    df["above_sma50"]  = (close > df["sma_50"]).astype(int)
    df["above_sma200"] = (close > df["sma_200"]).astype(int)
    df["price_sma20_pct"] = (close - df["sma_20"]) / (df["sma_20"] + 1e-9)
    df["price_sma50_pct"] = (close - df["sma_50"]) / (df["sma_50"] + 1e-9)
    hi252 = high.rolling(252).max()
    lo252 = low.rolling(252).min()
    df["52w_pct"]      = (close - lo252) / (hi252 - lo252 + 1e-9)
    df["golden_cross"] = ((df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))).astype(int)
    df["death_cross"]  = ((df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))).astype(int)

    # TARGET: 126 days (~6 months) — enough future data exists for all rows
    future_126 = close.shift(-126) / close - 1
    future_252 = close.shift(-252) / close - 1
    df["target_6mo"]   = (future_126 > 0.15).astype(int)   # >15% in 6 months
    df["target_1yr"]   = (future_252 > 0.20).astype(int)   # >20% in 1 year
    df["future_126"]   = future_126
    df["future_252"]   = future_252
    return df

all_dfs = []
for ticker in TICKERS:
    try:
        df = load_csv(ticker)
        df = build_features(df)
        df["ticker"] = ticker
        df.to_csv(f"data/features/{ticker}_features.csv")
        all_dfs.append(df)
        pos6  = df["target_6mo"].dropna().mean()*100
        pos1y = df["target_1yr"].dropna().mean()*100
        print(f"  {Fore.GREEN}✓ {ticker}: {len(df)} rows | 6mo wins: {pos6:.0f}% | 1yr wins: {pos1y:.0f}%")
    except Exception as e:
        print(f"  {Fore.RED}✗ {ticker}: {e}")

combined = pd.concat(all_dfs)
combined.to_csv("data/features/combined.csv")
print(f"\n{Fore.GREEN}✓ Combined: {len(combined):,} rows x {len(combined.columns)} features")
print(f"{Fore.GREEN}✓ Next: python ml/train_model.py")
