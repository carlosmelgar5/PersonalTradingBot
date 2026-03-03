"""
backend/app.py - Flask server that serves live LEAPS predictions
Run: python backend/app.py
Then open: http://localhost:5000
"""
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf, pandas as pd, numpy as np
import joblib, sqlite3, os, threading, time, warnings
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

TICKERS = ["NVDA","MSFT","GOOGL","META","AAPL","AMD","TSLA","AMZN",
           "XOM","CVX","LLY","MRNA","SPY","QQQ","IWM","XLE","XLK"]

FEATURES = ["rsi_14","rsi_7","macd","macd_signal","macd_hist","adx","adx_pos","adx_neg","cci",
    "bb_width","bb_pct","atr_pct","hv_20","hv_60","hv_90","mfi","vol_ratio","cmf",
    "returns_1d","returns_5d","returns_20d","returns_60d","gap","candle_body",
    "above_sma20","above_sma50","above_sma200","price_sma20_pct","price_sma50_pct",
    "52w_pct","stoch_k","stoch_d","roc_10","roc_30","williams","awesome",
    "golden_cross","death_cross","aroon_up","aroon_down"]

analyzer = SentimentIntensityAnalyzer()
cache = {"predictions": [], "market_mood": {}, "last_updated": None, "news": []}

def load_model():
    try:
        model  = joblib.load("models/ensemble.pkl")
        scaler = joblib.load("models/scaler.pkl")
        print("✓ Model loaded")
        return model, scaler
    except:
        print("⚠ No model found - run python ml/train_model.py first")
        return None, None

model, scaler = load_model()

def get_features(ticker):
    from datetime import timedelta
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 60: return None, None

    close, high, low, vol, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    df["sma_20"]  = ta.trend.sma_indicator(close, 20)
    df["sma_50"]  = ta.trend.sma_indicator(close, 50)
    df["sma_200"] = ta.trend.sma_indicator(close, 200)
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd(); df["macd_signal"] = macd.macd_signal(); df["macd_hist"] = macd.macd_diff()
    df["adx"] = ta.trend.adx(high,low,close); df["adx_pos"] = ta.trend.adx_pos(high,low,close); df["adx_neg"] = ta.trend.adx_neg(high,low,close)
    df["cci"] = ta.trend.cci(high,low,close)
    aroon = ta.trend.AroonIndicator(high=high,low=low,window=25)
    df["aroon_up"] = aroon.aroon_up(); df["aroon_down"] = aroon.aroon_down()
    df["rsi_14"] = ta.momentum.rsi(close,14); df["rsi_7"] = ta.momentum.rsi(close,7)
    df["stoch_k"] = ta.momentum.stoch(high,low,close); df["stoch_d"] = ta.momentum.stoch_signal(high,low,close)
    df["roc_10"] = ta.momentum.roc(close,10); df["roc_30"] = ta.momentum.roc(close,30)
    df["williams"] = ta.momentum.williams_r(high,low,close)
    df["awesome"] = ta.momentum.awesome_oscillator(high,low)
    bb = ta.volatility.BollingerBands(close)
    df["bb_width"] = bb.bollinger_wband(); df["bb_pct"] = bb.bollinger_pband()
    df["atr"] = ta.volatility.average_true_range(high,low,close)
    df["atr_pct"] = df["atr"] / (close+1e-9)
    df["hv_20"] = close.pct_change().rolling(20).std()*(252**0.5)
    df["hv_60"] = close.pct_change().rolling(60).std()*(252**0.5)
    df["hv_90"] = close.pct_change().rolling(90).std()*(252**0.5)
    df["obv"] = ta.volume.on_balance_volume(close,vol)
    df["mfi"] = ta.volume.money_flow_index(high,low,close,vol)
    df["vol_ratio"] = vol/(vol.rolling(20).mean()+1e-9)
    df["cmf"] = ta.volume.chaikin_money_flow(high,low,close,vol)
    df["returns_1d"]=close.pct_change(1); df["returns_5d"]=close.pct_change(5)
    df["returns_20d"]=close.pct_change(20); df["returns_60d"]=close.pct_change(60)
    df["gap"]=(open_-close.shift(1))/(close.shift(1)+1e-9)
    df["candle_body"]=(close-open_)/(open_+1e-9)
    df["above_sma20"]=(close>df["sma_20"]).astype(int)
    df["above_sma50"]=(close>df["sma_50"]).astype(int)
    df["above_sma200"]=(close>df["sma_200"]).astype(int)
    df["price_sma20_pct"]=(close-df["sma_20"])/(df["sma_20"]+1e-9)
    df["price_sma50_pct"]=(close-df["sma_50"])/(df["sma_50"]+1e-9)
    hi252=high.rolling(252).max(); lo252=low.rolling(252).min()
    df["52w_pct"]=(close-lo252)/(hi252-lo252+1e-9)
    df["golden_cross"]=((df["sma_50"]>df["sma_200"])&(df["sma_50"].shift(1)<=df["sma_200"].shift(1))).astype(int)
    df["death_cross"]=((df["sma_50"]<df["sma_200"])&(df["sma_50"].shift(1)>=df["sma_200"].shift(1))).astype(int)

    row = df.dropna(subset=FEATURES).iloc[-1]
    price = float(close.iloc[-1])
    change_pct = float((close.iloc[-1]/close.iloc[-2]-1)*100)
    return row[FEATURES].values.reshape(1,-1), {"price": price, "change_pct": change_pct,
        "rsi": float(row["rsi_14"]), "above_200": int(row["above_sma200"]),
        "hv_20": float(row["hv_20"]), "volume_ratio": float(row["vol_ratio"])}

def get_sentiment(ticker):
    try:
        import requests
        r = requests.get(f"https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&sort=hot&limit=10",
            headers={"User-Agent":"leaps_intel/1.0"}, timeout=5)
        posts = r.json()["data"]["children"]
        scores = [analyzer.polarity_scores(p["data"]["title"])["compound"] for p in posts]
        return round(float(np.mean(scores)),3) if scores else 0.0
    except: return 0.0

def run_predictions():
    while True:
        if model is None:
            time.sleep(60); continue
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running predictions...")
        predictions = []
        for ticker in TICKERS:
            try:
                X, info = get_features(ticker)
                if X is None: continue
                X_scaled = scaler.transform(X)
                conf = float(model.predict_proba(X_scaled)[0][1])
                sentiment = get_sentiment(ticker)
                price = info["price"]
                strike_call = round(price * 1.10 / 5) * 5
                signal = "BUY CALL" if conf >= 0.60 else "WATCH" if conf >= 0.50 else "HOLD"
                predictions.append({
                    "ticker": ticker,
                    "price": round(price, 2),
                    "change_pct": round(info["change_pct"], 2),
                    "confidence": round(conf * 100, 1),
                    "signal": signal,
                    "rsi": round(info["rsi"], 1),
                    "above_200sma": bool(info["above_200"]),
                    "sentiment": sentiment,
                    "sentiment_label": "Bullish" if sentiment > 0.05 else "Bearish" if sentiment < -0.05 else "Neutral",
                    "strike_suggestion": f"${strike_call}C Jan 2027",
                    "hv": round(info["hv_20"] * 100, 1),
                })
                print(f"  {ticker}: {conf*100:.1f}% → {signal}")
                time.sleep(0.3)
            except Exception as e:
                print(f"  {ticker} error: {e}")
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        cache["predictions"] = predictions
        cache["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cache["market_mood"] = {
            "avg_confidence": round(np.mean([p["confidence"] for p in predictions]),1) if predictions else 0,
            "buy_signals": sum(1 for p in predictions if p["signal"]=="BUY CALL"),
            "avg_sentiment": round(np.mean([p["sentiment"] for p in predictions]),3) if predictions else 0,
        }
        print(f"✓ Updated {len(predictions)} predictions")
        time.sleep(300)  # refresh every 5 minutes

@app.route("/api/predictions")
def api_predictions():
    return jsonify(cache)

@app.route("/")
def index():
    return render_template_string(HTML)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>LEAPS::INTEL</title>
<meta http-equiv="refresh" content="60">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Bebas+Neue&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#08090c;color:#e0e0d8;font-family:'IBM Plex Mono',monospace;font-size:13px}
.header{background:#0d0e12;border-bottom:1px solid #1e2028;padding:16px 28px;display:flex;justify-content:space-between;align-items:center}
.logo{font-family:'Bebas Neue';font-size:28px;letter-spacing:5px;color:#ffaa00}
.subtitle{color:#444;font-size:11px;letter-spacing:2px}
.timestamp{color:#ffaa00;font-size:11px}
.mood-bar{background:#0d0e12;border-bottom:1px solid #1e2028;padding:12px 28px;display:flex;gap:40px}
.mood-item{text-align:center}
.mood-label{color:#444;font-size:10px;letter-spacing:2px}
.mood-val{font-size:20px;font-weight:700;margin-top:4px}
.container{padding:24px 28px;max-width:1400px}
.section-title{color:#555;font-size:10px;letter-spacing:3px;margin-bottom:16px}
table{width:100%;border-collapse:collapse}
th{background:#0d0e12;color:#444;font-size:10px;letter-spacing:2px;padding:10px 14px;text-align:left;border-bottom:1px solid #1e2028}
td{padding:12px 14px;border-bottom:1px solid #111318;font-size:12px}
tr:hover{background:#0d0e12}
.signal-buy{color:#00ff88;font-weight:700}
.signal-watch{color:#ffaa00;font-weight:700}
.signal-hold{color:#444}
.bull{color:#00ff88}.bear{color:#ff4466}.neutral{color:#ffaa00}
.conf-bar{height:3px;background:#1e2028;border-radius:2px;margin-top:4px}
.conf-fill{height:100%;background:linear-gradient(90deg,#ffaa00,#00ff88);border-radius:2px}
.up{color:#00ff88}.down{color:#ff4466}
.disclaimer{color:#333;font-size:10px;padding:20px 28px;border-top:1px solid #111}
.refresh{color:#444;font-size:10px;text-align:right;padding:8px 28px}
</style>
<script>
async function loadData(){
  const r = await fetch("/api/predictions");
  const d = await r.json();
  const mood = d.market_mood || {};
  document.getElementById("buy-signals").textContent = mood.buy_signals || 0;
  document.getElementById("avg-conf").textContent = (mood.avg_confidence||0).toFixed(1)+"%";
  document.getElementById("avg-sent").textContent = mood.avg_sentiment > 0.05 ? "BULLISH" : mood.avg_sentiment < -0.05 ? "BEARISH" : "NEUTRAL";
  document.getElementById("updated").textContent = d.last_updated || "Loading...";
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  (d.predictions||[]).forEach(p => {
    const sigClass = p.signal==="BUY CALL" ? "signal-buy" : p.signal==="WATCH" ? "signal-watch" : "signal-hold";
    const chgClass = p.change_pct >= 0 ? "up" : "down";
    const sentClass = p.sentiment_label==="Bullish" ? "bull" : p.sentiment_label==="Bearish" ? "bear" : "neutral";
    tbody.innerHTML += `<tr>
      <td><b style="color:#ffaa00;font-size:14px">${p.ticker}</b></td>
      <td>$${p.price}</td>
      <td class="${chgClass}">${p.change_pct>=0?"+":""}${p.change_pct}%</td>
      <td><span class="${sigClass}">${p.signal}</span></td>
      <td>
        <div>${p.confidence}%</div>
        <div class="conf-bar"><div class="conf-fill" style="width:${p.confidence}%"></div></div>
      </td>
      <td style="color:${p.rsi>70?"#ff4466":p.rsi<30?"#00ff88":"#e0e0d8"}">${p.rsi}</td>
      <td>${p.above_200sma ? '<span class="bull">▲ YES</span>' : '<span class="bear">▼ NO</span>'}</td>
      <td class="${sentClass}">${p.sentiment_label} (${p.sentiment})</td>
      <td style="color:#ffaa00">${p.strike_suggestion}</td>
      <td style="color:#555">${p.hv}%</td>
    </tr>`;
  });
}
loadData();
setInterval(loadData, 30000);
</script>
</head>
<body>
<div class="header">
  <div>
    <div class="logo">LEAPS::INTEL</div>
    <div class="subtitle">ML-POWERED OPTIONS SCANNER — ROBINHOOD OPTIMIZED</div>
  </div>
  <div class="timestamp">Last updated: <span id="updated">Loading...</span><br>
  <span style="color:#00ff88;font-size:10px">● AUTO-REFRESHES EVERY 5 MIN</span></div>
</div>
<div class="mood-bar">
  <div class="mood-item"><div class="mood-label">BUY SIGNALS</div><div class="mood-val" id="buy-signals" style="color:#00ff88">-</div></div>
  <div class="mood-item"><div class="mood-label">AVG CONFIDENCE</div><div class="mood-val" id="avg-conf" style="color:#ffaa00">-</div></div>
  <div class="mood-item"><div class="mood-label">MARKET SENTIMENT</div><div class="mood-val" id="avg-sent" style="color:#ffaa00">-</div></div>
  <div class="mood-item"><div class="mood-label">TICKERS SCANNED</div><div class="mood-val" style="color:#888">17</div></div>
</div>
<div class="container">
  <div class="section-title">// LIVE LEAPS PREDICTIONS — ML ENSEMBLE MODEL</div>
  <table>
    <thead><tr>
      <th>TICKER</th><th>PRICE</th><th>TODAY</th><th>SIGNAL</th>
      <th>ML CONFIDENCE</th><th>RSI</th><th>ABOVE 200SMA</th>
      <th>REDDIT SENTIMENT</th><th>LEAPS SUGGESTION</th><th>HIST VOL</th>
    </tr></thead>
    <tbody id="tbody"><tr><td colspan="10" style="text-align:center;color:#444;padding:40px">
      Loading predictions... (first load takes ~2 min)
    </td></tr></tbody>
  </table>
</div>
<div class="disclaimer">⚠ NOT FINANCIAL ADVICE. For educational purposes only. Always do your own research. Options trading involves significant risk of loss.</div>
<div class="refresh">Page auto-refreshes every 60 seconds | API refreshes every 5 minutes</div>
</body></html>"""

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  LEAPS::INTEL — Live Web Dashboard       ║")
    print("║  Open: http://localhost:5000             ║")
    print("╚══════════════════════════════════════════╝")
    t = threading.Thread(target=run_predictions, daemon=True)
    t.start()
    app.run(debug=False, port=8080, host="0.0.0.0")
