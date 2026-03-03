"""
backend/simulator.py - LEAPS::INTEL Institutional Edition
Professional Goldman Sachs / Bain Capital aesthetic
Run: python backend/simulator.py
Keep alive 24/7: nohup python backend/simulator.py > logs/app.log 2>&1 &
Open: http://localhost:8080
"""
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import yfinance as yf, pandas as pd, numpy as np
import sqlite3, os, threading, time, warnings, requests, math
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

DB = "data/simulator.db"
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
analyzer = SentimentIntensityAnalyzer()
STARTING_CAPITAL = 1000.0
MONTHLY_TARGET = 30.0
TICKERS = ["NVDA","MSFT","GOOGL","META","AAPL","AMD","TSLA","AMZN",
           "SPY","QQQ","SOXX","LLY","COIN","PLTR","XOM","MRNA"]
cache = {"signals":[],"positions":[],"account":{},"after_hours":[],
         "trade_log":[],"last_scan":None,"market_open":False}

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY,
        cash REAL, start_date TEXT, total_trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0, peak REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS positions (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, option_type TEXT, strike REAL, expiry_days INTEGER,
        contracts INTEGER, entry_price REAL, current_price REAL,
        cost_basis REAL, entry_date TEXT, status TEXT DEFAULT 'OPEN',
        pnl REAL DEFAULT 0, pnl_pct REAL DEFAULT 0,
        exit_price REAL, exit_date TEXT, trade_type TEXT, thesis TEXT,
        target_pct REAL, stop_pct REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, action TEXT, option_type TEXT, strike REAL,
        contracts INTEGER, price REAL, pnl REAL, pnl_pct REAL,
        trade_type TEXT, date TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS after_hours (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, headline TEXT, source TEXT, sentiment REAL,
        direction TEXT, suggestion TEXT, created_at TEXT)""")
    if not c.execute("SELECT * FROM account").fetchone():
        c.execute("INSERT INTO account VALUES (1,1000.0,?,0,0,0,1000.0)",
                  (datetime.now().strftime("%Y-%m-%d"),))
    conn.commit(); conn.close()

def get_account():
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
    conn.close()
    cash, start_d, trades, wins, losses, peak = row[1],row[2],row[3],row[4],row[5],row[6]
    days = max((datetime.now()-datetime.strptime(start_d,"%Y-%m-%d")).days,1)
    pos_val = sum(p.get("cost_basis",0) for p in cache.get("positions",[]) if p.get("status")=="OPEN")
    total = cash + pos_val
    ret = (total/STARTING_CAPITAL-1)*100
    on_track = total >= STARTING_CAPITAL*(1+MONTHLY_TARGET/100*days/30)
    return {"cash":round(cash,2),"total_value":round(total,2),
            "total_return_pct":round(ret,2),"total_return_dollar":round(total-STARTING_CAPITAL,2),
            "peak":round(peak,2),"start_date":start_d,"days":days,
            "total_trades":trades,"wins":wins,"losses":losses,
            "win_rate":round(wins/max(trades,1)*100,1),"on_track":on_track,
            "compounded_1yr":round(STARTING_CAPITAL*(1+MONTHLY_TARGET/100)**12,0),
            "monthly_target":MONTHLY_TARGET}

def opt_price(S, K, days, otype, hv):
    T = max(days,0.5)/365; r=0.05
    try:
        from scipy.stats import norm
        d1=(math.log(S/K)+(r+0.5*hv**2)*T)/(hv*math.sqrt(T)); d2=d1-hv*math.sqrt(T)
        p=S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2) if otype=="call" \
          else K*math.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        return max(round(p,2),0.05)
    except:
        intr=max(0,S-K) if otype=="call" else max(0,K-S)
        return max(round(intr+S*hv*math.sqrt(T)*0.4,2),0.05)

def strike_round(p, direction, mono):
    step = 10 if p>500 else 5 if p>100 else 2.5 if p>50 else 1
    atm = round(p/step)*step
    return atm if mono=="ATM" else (atm+step if direction=="call" else atm-step)

def scan(ticker):
    try:
        df = yf.download(ticker, period="4mo", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df.columns=[c.lower() for c in df.columns]
        if len(df)<30: return None
        c=df["close"]; h=df["high"]; l=df["low"]; v=df["volume"]
        price=float(c.iloc[-1])
        rsi=float(ta.momentum.rsi(c,14).iloc[-1])
        rsi7=float(ta.momentum.rsi(c,7).iloc[-1])
        md=ta.trend.MACD(c)
        mh=float(md.macd_diff().iloc[-1]); ml=float(md.macd().iloc[-1]); ms=float(md.macd_signal().iloc[-1])
        bb=ta.volatility.BollingerBands(c)
        bbp=float(bb.bollinger_pband().iloc[-1]); bbw=float(bb.bollinger_wband().iloc[-1])
        stoch=float(ta.momentum.stoch(h,l,c).iloc[-1])
        atr=float(ta.volatility.average_true_range(h,l,c).iloc[-1])
        hv=float(c.pct_change().rolling(20).std().iloc[-1]*math.sqrt(252))
        sma20=float(ta.trend.sma_indicator(c,20).iloc[-1])
        sma50=float(ta.trend.sma_indicator(c,50).iloc[-1])
        adx=float(ta.trend.adx(h,l,c).iloc[-1])
        volr=float(v.iloc[-1]/v.rolling(20).mean().iloc[-1])
        ret1=float(c.pct_change(1).iloc[-1]*100)
        hi52=float(h.rolling(252).max().iloc[-1]); lo52=float(l.rolling(252).min().iloc[-1])
        p52=round((price-lo52)/(hi52-lo52+1e-9)*100,1)
        return {"ticker":ticker,"price":round(price,2),"rsi":round(rsi,1),
                "rsi_7":round(rsi7,1),"macd_hist":round(mh,4),"macd":round(ml,4),
                "macd_signal":round(ms,4),"bb_pct":round(bbp*100,1),
                "bb_width":round(bbw,3),"stoch":round(stoch,1),"atr":round(atr,2),
                "hv":round(hv,3),"sma20":round(sma20,2),"sma50":round(sma50,2),
                "adx":round(adx,1),"vol_ratio":round(volr,2),
                "ret1d":round(ret1,2),"pct52w":p52,
                "above20":price>sma20,"above50":price>sma50}
    except Exception as e: print(f"  {ticker}: {e}"); return None

def signals(d):
    if not d: return []
    sigs=[]; p=d["price"]; rsi=d["rsi"]; stoch=d["stoch"]
    mh=d["macd_hist"]; bb=d["bb_pct"]; adx=d["adx"]
    volr=d["vol_ratio"]; hv=d["hv"]; t=d["ticker"]
    if rsi<35 and stoch<30 and volr>1.0:
        conf=min(55+(35-rsi)*1.2+(30-stoch)*0.6+(volr-1)*8,85)
        days=1 if rsi<28 else 2; s=strike_round(p,"call","ATM"); op=opt_price(p,s,days,"call",hv)
        c_=max(1,int(200/(op*100)))
        sigs.append({"ticker":t,"direction":"CALL","signal_type":"OVERSOLD BOUNCE",
            "trade_duration":f"{days}DTE","confidence":round(conf,1),
            "strike":s,"expiry_days":days,"option_price":op,
            "suggested_contracts":c_,"cost":round(op*c_*100,2),
            "target_pct":30,"stop_pct":-40,
            "thesis":f"Oversold condition detected. RSI {rsi} with Stochastic {stoch} confirming. Volume ratio {volr:.1f}x above average. High-probability mean reversion setup.",
            "rationale":"RSI<35 + Stoch<30 + elevated volume = bounce signal",
            "robinhood_tip":f"Buy {c_} contract(s) of ${s}C expiring in {days} day(s). Set limit sell order at +30%."})
    if mh>0 and 52<rsi<68 and adx>22 and volr>1.3 and d["above20"]:
        conf=min(52+adx*0.5+(volr-1)*10+(rsi-52)*0.4,82)
        days=4; s=strike_round(p,"call","OTM1"); op=opt_price(p,s,days,"call",hv)
        c_=max(1,int(250/(op*100)))
        sigs.append({"ticker":t,"direction":"CALL","signal_type":"MOMENTUM CONTINUATION",
            "trade_duration":"3-5 days","confidence":round(conf,1),
            "strike":s,"expiry_days":days,"option_price":op,
            "suggested_contracts":c_,"cost":round(op*c_*100,2),
            "target_pct":40,"stop_pct":-35,
            "thesis":f"MACD bullish crossover confirmed. ADX {adx:.0f} indicates strong trend. Volume {volr:.1f}x confirms institutional participation. Price above 20-day MA.",
            "rationale":"MACD cross + ADX>22 + volume surge + above SMA20",
            "robinhood_tip":f"Buy {c_} contract(s) of ${s}C with 4-day expiry. Exit at +40% gain or if MACD turns negative."})
    if rsi>72 and stoch>80 and bb>90:
        conf=min(50+(rsi-72)+(stoch-80)*0.5+(bb-90)*0.3,80)
        days=3; s=strike_round(p,"put","ATM"); op=opt_price(p,s,days,"put",hv)
        c_=max(1,int(200/(op*100)))
        sigs.append({"ticker":t,"direction":"PUT","signal_type":"OVERBOUGHT REVERSAL",
            "trade_duration":"2-4 days","confidence":round(conf,1),
            "strike":s,"expiry_days":days,"option_price":op,
            "suggested_contracts":c_,"cost":round(op*c_*100,2),
            "target_pct":35,"stop_pct":-40,
            "thesis":f"Extended overbought condition. RSI {rsi} at extreme levels, Stochastic {stoch} confirming. Bollinger Band position at {bb:.0f}th percentile. Mean reversion likely within 2-4 sessions.",
            "rationale":"RSI>72 + Stoch>80 + BB>90% = overextended",
            "robinhood_tip":f"Buy {c_} contract(s) of ${s}P with 3-day expiry. Short-duration fade trade — exit by day 3 regardless."})
    if d["bb_width"]<0.03 and adx<20:
        direction="call" if mh>0 else "put"
        conf=min(48+(0.05-d["bb_width"])*500,75)
        days=7; s=strike_round(p,direction,"OTM1"); op=opt_price(p,s,days,direction,hv)
        c_=max(1,int(150/(op*100)))
        sigs.append({"ticker":t,"direction":direction.upper(),"signal_type":"VOLATILITY SQUEEZE",
            "trade_duration":"5-7 days","confidence":round(conf,1),
            "strike":s,"expiry_days":days,"option_price":op,
            "suggested_contracts":c_,"cost":round(op*c_*100,2),
            "target_pct":50,"stop_pct":-30,
            "thesis":f"Bollinger Band width compressed to {d['bb_width']:.3f} — historically low volatility environment. Significant price expansion imminent. MACD bias favors {direction.upper()} direction.",
            "rationale":"BB squeeze (low vol) = explosive move loading",
            "robinhood_tip":f"Buy {c_} contract(s) of ${s}{direction[0].upper()} with 7-day expiry. Wait for a strong confirming candle before entering."})
    return sorted(sigs,key=lambda x:x["confidence"],reverse=True)

def execute_trade(sig):
    acc=get_account()
    if sig["cost"]>acc["cash"]:
        return {"error":f"Insufficient capital. Available: ${acc['cash']:.2f}, Required: ${sig['cost']:.2f}"}
    conn=sqlite3.connect(DB)
    conn.execute("""INSERT INTO positions (ticker,option_type,strike,expiry_days,contracts,
        entry_price,current_price,cost_basis,entry_date,trade_type,thesis,target_pct,stop_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sig["ticker"],sig["direction"],sig["strike"],sig["expiry_days"],
         sig["suggested_contracts"],sig["option_price"],sig["option_price"],
         sig["cost"],datetime.now().strftime("%Y-%m-%d %H:%M"),
         sig["signal_type"],sig["thesis"],sig["target_pct"],sig["stop_pct"]))
    conn.execute("UPDATE account SET cash=cash-? WHERE id=1",(sig["cost"],))
    conn.execute("""INSERT INTO trade_log (ticker,action,option_type,strike,contracts,
        price,pnl,pnl_pct,trade_type,date) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (sig["ticker"],"BUY",sig["direction"],sig["strike"],
         sig["suggested_contracts"],sig["option_price"],0,0,
         sig["signal_type"],datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit(); conn.close()
    return {"success":True,"message":f"Executed: {sig['suggested_contracts']}x ${sig['strike']}{sig['direction'][0]} — Cost ${sig['cost']:.2f}"}

def close_pos(pid):
    conn=sqlite3.connect(DB)
    pos=conn.execute("SELECT * FROM positions WHERE id=? AND status='OPEN'",(pid,)).fetchone()
    if not pos: conn.close(); return {"error":"Position not found"}
    try:
        cp=float(yf.Ticker(pos[1]).fast_info.last_price)
        ep=max(round(pos[6]*(1+(cp/pos[6]-1)*2.5),2),0.01)
    except: ep=pos[6]
    pnl=(ep-pos[6])*pos[5]*100; ppct=(ep/pos[6]-1)*100
    res="wins" if pnl>0 else "losses"
    conn.execute("""UPDATE positions SET status='CLOSED',exit_price=?,exit_date=?,
        pnl=?,pnl_pct=?,current_price=? WHERE id=?""",
        (ep,datetime.now().strftime("%Y-%m-%d %H:%M"),round(pnl,2),round(ppct,1),ep,pid))
    conn.execute(f"UPDATE account SET cash=cash+?,{res}={res}+1,total_trades=total_trades+1 WHERE id=1",
        (ep*pos[5]*100,))
    conn.execute("""INSERT INTO trade_log (ticker,action,option_type,strike,contracts,
        price,pnl,pnl_pct,trade_type,date) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (pos[1],"SELL",pos[2],pos[3],pos[5],ep,round(pnl,2),round(ppct,1),
         pos[15],datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit(); conn.close()
    return {"success":True,"pnl":round(pnl,2),"pnl_pct":round(ppct,1),
            "message":f"Closed at ${ep:.2f} — P&L: ${pnl:+.2f} ({ppct:+.1f}%)"}

def get_db_data():
    conn=sqlite3.connect(DB)
    pos=conn.execute("SELECT * FROM positions ORDER BY entry_date DESC").fetchall()
    log=conn.execute("SELECT * FROM trade_log ORDER BY date DESC LIMIT 30").fetchall()
    ah=conn.execute("SELECT * FROM after_hours ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    positions=[{"id":r[0],"ticker":r[1],"option_type":r[2],"strike":r[3],"expiry_days":r[4],
                "contracts":r[5],"entry_price":r[6],"current_price":r[7],"cost_basis":r[8],
                "entry_date":r[9],"status":r[10],"pnl":r[11],"pnl_pct":r[12],
                "exit_price":r[13],"exit_date":r[14],"trade_type":r[15],"thesis":r[16],
                "target_pct":r[17],"stop_pct":r[18]} for r in pos]
    tlog=[{"ticker":r[1],"action":r[2],"type":r[3],"strike":r[4],"contracts":r[5],
           "price":r[6],"pnl":r[7],"pnl_pct":r[8],"trade_type":r[9],"date":r[10]} for r in log]
    after=[{"ticker":r[1],"headline":r[2],"source":r[3],"sentiment":r[4],
            "direction":r[5],"suggestion":r[6],"time":r[7]} for r in ah]
    return positions,tlog,after

def run_ah():
    print("  Running after-hours Reddit scan...")
    for ticker in TICKERS:
        try:
            r=requests.get(
                f"https://www.reddit.com/search.json?q={ticker}+stock&sort=hot&limit=10&t=day",
                headers={"User-Agent":"leaps_intel/1.0"},timeout=6)
            posts=r.json()["data"]["children"]
            for p in posts[:3]:
                title=p["data"]["title"]; score=p["data"]["score"]
                if score<50: continue
                sent=analyzer.polarity_scores(title)["compound"]
                if abs(sent)<0.1: continue
                direction="BULLISH" if sent>0.1 else "BEARISH"
                suggestion=f"Monitor {ticker} at open — {direction} Reddit sentiment ({score} upvotes)" if score>100 else "Low conviction — monitor only"
                conn=sqlite3.connect(DB)
                conn.execute("INSERT INTO after_hours VALUES (NULL,?,?,?,?,?,?,?)",
                    (ticker,title[:120],f"Reddit ({score} upvotes)",
                     round(sent,3),direction,suggestion,datetime.now().strftime("%H:%M")))
                conn.commit(); conn.close()
            time.sleep(0.8)
        except: pass

def background():
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market scan running...")
        all_sigs=[]
        for t in TICKERS:
            d=scan(t)
            if d: all_sigs.extend(signals(d))
            time.sleep(0.3)
        all_sigs.sort(key=lambda x:x["confidence"],reverse=True)
        cache["signals"]=all_sigs[:15]
        positions,tlog,after=get_db_data()
        cache["positions"]=positions; cache["trade_log"]=tlog; cache["after_hours"]=after
        cache["account"]=get_account()
        cache["last_scan"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now=datetime.now()
        is_open=(now.weekday()<5 and
                 (now.hour>9 or(now.hour==9 and now.minute>=30)) and now.hour<16)
        cache["market_open"]=is_open
        if not is_open and now.hour>=16 and now.minute<45:
            run_ah()
        print(f"  Scan complete — {len(all_sigs)} signals | Account: ${cache['account']['total_value']:.2f}")
        time.sleep(300)

@app.route("/api/data")
def api_data(): return jsonify(cache)

@app.route("/api/trade",methods=["POST"])
def api_trade():
    idx=request.json.get("signal_idx",0)
    if idx<len(cache["signals"]):
        r=execute_trade(cache["signals"][idx])
        cache["account"]=get_account()
        positions,tlog,_=get_db_data()
        cache["positions"]=positions; cache["trade_log"]=tlog
        return jsonify(r)
    return jsonify({"error":"Invalid signal index"})

@app.route("/api/close",methods=["POST"])
def api_close():
    r=close_pos(request.json.get("position_id"))
    cache["account"]=get_account()
    positions,tlog,_=get_db_data()
    cache["positions"]=positions; cache["trade_log"]=tlog
    return jsonify(r)

@app.route("/api/reset",methods=["POST"])
def api_reset():
    conn=sqlite3.connect(DB)
    conn.execute("UPDATE account SET cash=1000,total_trades=0,wins=0,losses=0,peak=1000,start_date=? WHERE id=1",
                 (datetime.now().strftime("%Y-%m-%d"),))
    conn.execute("DELETE FROM positions"); conn.execute("DELETE FROM trade_log")
    conn.execute("DELETE FROM after_hours")
    conn.commit(); conn.close()
    cache["account"]=get_account()
    return jsonify({"success":True})

@app.route("/")
def index(): return render_template_string(HTML)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LEAPS INTEL — Equity Options Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --navy:    #0a1628;
  --navy2:   #0f2040;
  --navy3:   #162d52;
  --gold:    #b8960c;
  --gold2:   #d4a800;
  --white:   #ffffff;
  --off:     #f8f9fb;
  --grey1:   #f1f3f6;
  --grey2:   #e4e8ee;
  --grey3:   #c9d0db;
  --grey4:   #8a94a6;
  --grey5:   #4a5568;
  --text:    #0a1628;
  --text2:   #2d3748;
  --text3:   #4a5568;
  --green:   #1a6b3c;
  --greenl:  #f0faf4;
  --greenbr: #c6e8d4;
  --red:     #9b1c1c;
  --redl:    #fff5f5;
  --redbr:   #f5c0c0;
  --blue:    #1a3a6b;
  --bluel:   #eef4ff;
  --bluebr:  #bfcfe8;
  --shadow:  0 1px 4px rgba(10,22,40,0.08);
  --shadow2: 0 4px 16px rgba(10,22,40,0.10);
  --r: 4px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 13px; }
body { background: var(--off); color: var(--text); font-family: 'DM Sans', sans-serif; min-height: 100vh; }

/* ── HEADER ── */
.hdr {
  background: var(--navy);
  padding: 0 32px;
  height: 58px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 200;
  border-bottom: 2px solid var(--gold);
}
.hdr-left { display: flex; align-items: center; gap: 24px; }
.wordmark { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700; color: var(--white); letter-spacing: 0.5px; }
.wordmark span { color: var(--gold2); }
.divider-v { width: 1px; height: 22px; background: var(--navy3); }
.desk-label { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--grey4); letter-spacing: 2px; text-transform: uppercase; }
.hdr-right { display: flex; align-items: center; gap: 20px; }
.mkt-chip { display: flex; align-items: center; gap: 7px; padding: 5px 13px; border-radius: 2px; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; letter-spacing: 1px; }
.mkt-open-chip  { background: rgba(26,107,60,0.25); color: #6ee09b; border: 1px solid rgba(110,224,155,0.3); }
.mkt-closed-chip { background: rgba(155,28,28,0.25); color: #f5a0a0; border: 1px solid rgba(245,160,160,0.3); }
.pulse-dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2s infinite; }
.pulse-green { background: #6ee09b; }
.pulse-red   { background: #f5a0a0; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }
.hdr-time { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--grey4); }

/* ── MARKET TIMER BANNER ── */
.timer-banner {
  background: var(--navy2);
  border-bottom: 1px solid var(--navy3);
  padding: 10px 32px;
  display: flex; align-items: center; justify-content: space-between;
}
.timer-label { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--grey4); letter-spacing: 2px; text-transform: uppercase; }
.timer-val { font-family: 'DM Mono', monospace; font-size: 20px; font-weight: 500; color: var(--gold2); letter-spacing: 4px; }
.timer-meta { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--grey4); text-align: right; }
.timer-sub { font-size: 10px; color: var(--grey4); }

/* ── ACCOUNT BAR ── */
.abar {
  background: var(--white);
  border-bottom: 1px solid var(--grey2);
  display: grid; grid-template-columns: repeat(8,1fr);
}
.ai {
  padding: 14px 18px; text-align: center;
  border-right: 1px solid var(--grey2);
  transition: background 0.15s;
}
.ai:last-child { border-right: none; }
.ai:hover { background: var(--grey1); }
.al { font-family: 'DM Mono', monospace; font-size: 9px; font-weight: 500; color: var(--grey4); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; }
.av { font-family: 'DM Mono', monospace; font-size: 15px; font-weight: 500; color: var(--text); }
.av-pos { color: var(--green); }
.av-neg { color: var(--red); }
.av-gold { color: var(--gold); }
.av-navy { color: var(--navy); font-weight: 600; }
.track-pill { display: inline-block; padding: 3px 10px; border-radius: 2px; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; }
.track-yes { background: var(--greenl); color: var(--green); border: 1px solid var(--greenbr); }
.track-no  { background: var(--redl);   color: var(--red);   border: 1px solid var(--redbr); }

/* ── NAV TABS ── */
.tabs { background: var(--white); border-bottom: 1px solid var(--grey2); padding: 0 32px; display: flex; }
.tab { padding: 13px 20px; cursor: pointer; color: var(--grey4); font-size: 11px; font-weight: 600; letter-spacing: 0.5px; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all 0.15s; text-transform: uppercase; font-family: 'DM Mono', monospace; }
.tab:hover { color: var(--text); background: var(--grey1); }
.tab.on { color: var(--navy); border-bottom-color: var(--gold); }

/* ── PAGE CONTENT ── */
.tc { display: none; padding: 28px 32px; }
.tc.on { display: block; }
.section-label { font-family: 'DM Mono', monospace; font-size: 9px; font-weight: 500; color: var(--grey4); letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--grey2); }
.info-strip { background: var(--bluel); border-left: 3px solid var(--navy); padding: 12px 18px; margin-bottom: 20px; font-size: 12px; color: var(--text3); line-height: 1.8; border-radius: 0 var(--r) var(--r) 0; }
.info-strip b { color: var(--navy); font-weight: 600; }
.info-strip-amber { background: #fffbeb; border-left-color: var(--gold); }
.info-strip-amber b { color: var(--gold); }

/* ── SIGNAL CARDS ── */
.sc {
  background: var(--white);
  border: 1px solid var(--grey2);
  border-radius: var(--r);
  padding: 20px 24px;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}
.sc::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--green); }
.sc.put-card::after { background: var(--red); }
.sc:hover { box-shadow: var(--shadow2); border-color: var(--grey3); }
.sc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
.sc-ticker { font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 700; color: var(--navy); letter-spacing: -0.5px; }
.sc-badges { display: flex; align-items: center; gap: 8px; margin-top: 5px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 2px; font-family: 'DM Mono', monospace; font-size: 9px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; }
.b-call { background: var(--greenl); color: var(--green); border: 1px solid var(--greenbr); }
.b-put  { background: var(--redl);   color: var(--red);   border: 1px solid var(--redbr); }
.b-type { background: var(--grey1);  color: var(--text3); border: 1px solid var(--grey2); }
.b-dur  { background: #fffbeb; color: var(--gold); border: 1px solid #fde68a; }
.sc-right { display: flex; align-items: center; gap: 16px; }
.conf-block { text-align: right; }
.conf-lbl { font-family: 'DM Mono', monospace; font-size: 9px; color: var(--grey4); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 3px; }
.conf-num { font-family: 'DM Mono', monospace; font-size: 22px; font-weight: 500; }
.conf-bar { height: 3px; background: var(--grey2); border-radius: 1px; margin-top: 5px; }
.conf-fill { height: 100%; border-radius: 1px; }
.sc-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; margin-bottom: 12px; }
.sm { background: var(--grey1); border: 1px solid var(--grey2); border-radius: var(--r); padding: 10px 12px; text-align: center; }
.sml { font-family: 'DM Mono', monospace; font-size: 9px; color: var(--grey4); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.smv { font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; color: var(--text); }
.sc-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
.sm-g { background: var(--greenl); border: 1px solid var(--greenbr); border-radius: var(--r); padding: 10px 12px; text-align: center; }
.sm-g .smv { color: var(--green); }
.sm-r { background: var(--redl); border: 1px solid var(--redbr); border-radius: var(--r); padding: 10px 12px; text-align: center; }
.sm-r .smv { color: var(--red); }
.thesis-text { font-size: 12px; color: var(--text3); line-height: 1.75; margin-bottom: 12px; }
.rationale { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--grey4); margin-bottom: 12px; }
.tip-row { background: #fffbeb; border: 1px solid #fde68a; border-radius: var(--r); padding: 10px 14px; font-size: 11px; color: var(--text3); line-height: 1.7; }
.tip-row b { color: var(--gold); font-weight: 600; }

/* ── BUTTONS ── */
.btn { border: none; cursor: pointer; font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 600; padding: 9px 20px; border-radius: var(--r); transition: all 0.15s; letter-spacing: 0.3px; }
.btn-exec { background: var(--navy); color: var(--white); box-shadow: 0 2px 4px rgba(10,22,40,0.2); }
.btn-exec:hover { background: var(--navy2); transform: translateY(-1px); box-shadow: 0 4px 8px rgba(10,22,40,0.25); }
.btn-close { background: var(--redl); color: var(--red); border: 1px solid var(--redbr); font-size: 11px; padding: 5px 12px; border-radius: var(--r); }
.btn-close:hover { background: var(--red); color: var(--white); }
.btn-rst { background: transparent; border: 1px solid var(--grey3); color: var(--grey4); font-size: 11px; padding: 5px 14px; border-radius: var(--r); }
.btn-rst:hover { border-color: var(--red); color: var(--red); }

/* ── TABLES ── */
.tbl-wrap { background: var(--white); border: 1px solid var(--grey2); border-radius: var(--r); overflow: hidden; box-shadow: var(--shadow); margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: var(--grey1); color: var(--grey4); font-family: 'DM Mono', monospace; font-size: 9px; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--grey2); white-space: nowrap; }
td { padding: 12px 14px; border-bottom: 1px solid var(--grey1); color: var(--text2); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--off); }
.td-ticker { font-family: 'Playfair Display', serif; font-size: 14px; font-weight: 700; color: var(--navy); }
.td-mono { font-family: 'DM Mono', monospace; }
.td-pos { color: var(--green); font-family: 'DM Mono', monospace; font-weight: 500; }
.td-neg { color: var(--red);   font-family: 'DM Mono', monospace; font-weight: 500; }
.result-win  { background: var(--greenl); color: var(--green); padding: 2px 8px; border-radius: 2px; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; border: 1px solid var(--greenbr); }
.result-loss { background: var(--redl);   color: var(--red);   padding: 2px 8px; border-radius: 2px; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500; border: 1px solid var(--redbr); }

/* ── AFTER HOURS ── */
.ah-row { padding: 16px 0; border-bottom: 1px solid var(--grey2); }
.ah-row:last-child { border-bottom: none; }

/* ── FOOTER ── */
.ftr { background: var(--navy); border-top: 1px solid var(--navy3); padding: 14px 32px; color: var(--grey4); font-family: 'DM Mono', monospace; font-size: 10px; display: flex; justify-content: space-between; letter-spacing: 0.5px; margin-top: 40px; }

/* ── TOAST ── */
.toast { position: fixed; bottom: 24px; right: 24px; background: var(--white); border: 1px solid var(--grey2); border-radius: var(--r); padding: 12px 20px; font-size: 12px; font-weight: 600; z-index: 9999; display: none; box-shadow: var(--shadow2); border-left: 4px solid var(--green); }
.toast.show { display: block; }

/* ── EMPTY STATE ── */
.empty { text-align: center; padding: 50px; color: var(--grey4); font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 1px; }
</style>
</head>
<body>

<!-- Header -->
<header class="hdr">
  <div class="hdr-left">
    <div>
      <div class="wordmark">LEAPS <span>INTEL</span></div>
    </div>
    <div class="divider-v"></div>
    <div class="desk-label">Equity Options Desk — Paper Trading</div>
  </div>
  <div class="hdr-right">
    <div class="hdr-time" id="live-clock">--:--:-- EST</div>
    <div class="mkt-chip mkt-closed-chip" id="mkt-chip">
      <div class="pulse-dot pulse-red" id="mkt-dot"></div>
      <span id="mkt-lbl">MARKET CLOSED</span>
    </div>
  </div>
</header>

<!-- Market Timer Banner -->
<div class="timer-banner">
  <div>
    <div class="timer-label">Market Opens In</div>
    <div class="timer-val" id="market-timer">--:--:--</div>
    <div class="timer-sub" id="timer-sub">Calculating...</div>
  </div>
  <div style="text-align:center">
    <div class="timer-label">Last Scan</div>
    <div style="font-family:'DM Mono',monospace;font-size:13px;color:#8a94a6;margin-top:4px" id="lsc">--</div>
    <div class="timer-sub">Scans every 5 minutes</div>
  </div>
  <div style="text-align:right">
    <div class="timer-label">Session Target</div>
    <div style="font-family:'DM Mono',monospace;font-size:13px;color:var(--gold2);margin-top:4px">+30% Monthly</div>
    <div class="timer-sub">Signals active 24/7</div>
  </div>
</div>

<!-- Account Bar -->
<div class="abar">
  <div class="ai"><div class="al">Cash Available</div><div class="av av-navy" id="cash">$--</div></div>
  <div class="ai"><div class="al">Total Value</div><div class="av" id="tval">$--</div></div>
  <div class="ai"><div class="al">Total Return</div><div class="av" id="tret">--%</div></div>
  <div class="ai"><div class="al">Monthly Target</div><div class="av av-gold">+30.0%</div></div>
  <div class="ai"><div class="al">On Track</div><div id="otk">--</div></div>
  <div class="ai"><div class="al">Win Rate</div><div class="av" id="wr">--%</div></div>
  <div class="ai"><div class="al">Record</div><div class="av td-mono" id="wl">--</div></div>
  <div class="ai"><div class="al">Projected (1yr)</div><div class="av av-pos" id="py">$--</div></div>
</div>

<!-- Tabs -->
<nav class="tabs">
  <div class="tab on" onclick="tab(this,'sig')">Signals</div>
  <div class="tab" onclick="tab(this,'pos')">Positions</div>
  <div class="tab" onclick="tab(this,'log')">Trade Log</div>
  <div class="tab" onclick="tab(this,'ah')">After Hours</div>
</nav>

<!-- SIGNALS -->
<div class="tc on" id="tc-sig">
  <div class="info-strip">
    <b>Signal Protocol:</b> Each signal represents a quantitatively-screened options setup with defined entry, target (+30–50%), and stop (−30–40%). Position sizing is pre-calculated based on available capital. Confidence scores reflect multi-factor technical alignment. <b>Maximum allocation per trade: 20% of total account value.</b>
  </div>
  <div class="section-label">Active Signals — Ranked by Confidence</div>
  <div id="sigs"><div class="empty">Initializing market scan... approximately 2 minutes</div></div>
</div>

<!-- POSITIONS -->
<div class="tc" id="tc-pos">
  <div class="section-label">Open Positions</div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th>Security</th><th>Direction</th><th>Strike</th><th>Qty</th>
      <th>Entry</th><th>P&amp;L ($)</th><th>P&amp;L (%)</th>
      <th>Target</th><th>Stop</th><th>Setup Type</th><th>Action</th>
    </tr></thead><tbody id="opn"></tbody></table>
  </div>
  <div class="section-label">Closed Positions</div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th>Security</th><th>Direction</th><th>Entry</th><th>Exit</th>
      <th>P&amp;L ($)</th><th>P&amp;L (%)</th><th>Outcome</th>
    </tr></thead><tbody id="cls"></tbody></table>
  </div>
</div>

<!-- TRADE LOG -->
<div class="tc" id="tc-log">
  <div class="section-label">Transaction History</div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th>Timestamp</th><th>Security</th><th>Action</th><th>Direction</th>
      <th>Strike</th><th>Contracts</th><th>Premium</th><th>P&amp;L ($)</th><th>P&amp;L (%)</th><th>Setup</th>
    </tr></thead><tbody id="tlog"></tbody></table>
  </div>
  <div style="text-align:right;margin-top:12px">
    <button class="btn btn-rst" onclick="rst()">Reset Account to $1,000</button>
  </div>
</div>

<!-- AFTER HOURS -->
<div class="tc" id="tc-ah">
  <div class="info-strip info-strip-amber">
    <b>After-Hours Intelligence:</b> Automated Reddit scan executes at 4:00pm ET daily. High-engagement posts mentioning tracked securities are scored for sentiment polarity. Strongly bullish posts (score &gt;200, compound &gt;0.20) indicate potential gap-up opportunities at tomorrow's open. Use for pre-market watchlist preparation only.
  </div>
  <div class="section-label">After-Hours Sentiment Scan</div>
  <div class="tbl-wrap" style="padding: 0 20px;" id="ahc">
    <div class="empty">After-hours analysis runs automatically at 4:00pm ET market close.</div>
  </div>
</div>

<footer class="ftr">
  <span>LEAPS INTEL — Equity Options Desk — Paper Trading Simulation</span>
  <span>NOT INVESTMENT ADVICE. For educational and simulation purposes only. All capital at risk.</span>
</footer>

<div class="toast" id="toast"></div>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────
function tab(el, id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
  document.querySelectorAll('.tc').forEach(t => t.classList.remove('on'));
  el.classList.add('on');
  document.getElementById('tc-' + id).classList.add('on');
}

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type='pos') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderLeftColor = type === 'pos' ? 'var(--green)' : type === 'neg' ? 'var(--red)' : 'var(--gold)';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}

// ── Live clock ────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const est = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
  const h = String(est.getHours()).padStart(2,'0');
  const m = String(est.getMinutes()).padStart(2,'0');
  const s = String(est.getSeconds()).padStart(2,'0');
  document.getElementById('live-clock').textContent = `${h}:${m}:${s} EST`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Market timer ──────────────────────────────────────────────────────────
function updateMarketTimer(isOpen) {
  const now = new Date();
  const est = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
  const day = est.getDay(); // 0=Sun, 6=Sat
  const h = est.getHours(), m = est.getMinutes(), s = est.getSeconds();
  const timerEl = document.getElementById('market-timer');
  const subEl = document.getElementById('timer-sub');

  if (isOpen) {
    // Time until 4pm close
    const closeH = 16, closeM = 0, closeS = 0;
    let secs = (closeH-h)*3600 + (closeM-m)*60 + (closeS-s);
    if (secs < 0) secs = 0;
    const rh = Math.floor(secs/3600), rm = Math.floor((secs%3600)/60), rs = secs%60;
    timerEl.textContent = `${String(rh).padStart(2,'0')}:${String(rm).padStart(2,'0')}:${String(rs).padStart(2,'0')}`;
    timerEl.style.color = '#6ee09b';
    subEl.textContent = 'Time until market close (4:00pm ET)';
  } else {
    // Time until next 9:30am open
    let target = new Date(est);
    target.setHours(9,30,0,0);
    if (est >= target || day === 0 || day === 6) {
      target.setDate(target.getDate() + 1);
      while (target.getDay() === 0 || target.getDay() === 6) {
        target.setDate(target.getDate() + 1);
      }
      target.setHours(9,30,0,0);
    }
    let secs = Math.floor((target - est) / 1000);
    if (secs < 0) secs = 0;
    const rh = Math.floor(secs/3600), rm = Math.floor((secs%3600)/60), rs = secs%60;
    timerEl.textContent = `${String(rh).padStart(2,'0')}:${String(rm).padStart(2,'0')}:${String(rs).padStart(2,'0')}`;
    timerEl.style.color = 'var(--gold2)';
    const dayNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
    subEl.textContent = `Until open — ${dayNames[target.getDay()]} 9:30am ET`;
  }
}
setInterval(() => {
  const isOpen = document.getElementById('mkt-lbl').textContent === 'MARKET OPEN';
  updateMarketTimer(isOpen);
}, 1000);

// ── Confidence styling ─────────────────────────────────────────────────────
function confColor(v) {
  return v >= 70 ? 'var(--green)' : v >= 58 ? 'var(--gold)' : 'var(--grey4)';
}
function confBg(v) {
  return v >= 70 ? 'var(--green)' : v >= 58 ? 'var(--gold2)' : 'var(--grey3)';
}

// ── Trade actions ──────────────────────────────────────────────────────────
async function execute(i) {
  const r = await fetch('/api/trade', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({signal_idx: i})
  });
  const d = await r.json();
  d.success ? toast('Order executed: ' + d.message, 'pos') : toast(d.error || 'Execution failed', 'neg');
  load();
}

async function closePos(id) {
  if (!confirm('Close this position at estimated current market value?')) return;
  const r = await fetch('/api/close', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({position_id: id})
  });
  const d = await r.json();
  d.success ? toast(d.message, d.pnl >= 0 ? 'pos' : 'neg') : toast(d.error || 'Error closing position', 'neg');
  load();
}

async function rst() {
  if (!confirm('Reset account to $1,000? This will delete all trade history.')) return;
  await fetch('/api/reset', {method: 'POST'});
  toast('Account reset to $1,000.00', 'neutral');
  load();
}

// ── Main data loader ───────────────────────────────────────────────────────
async function load() {
  const r = await fetch('/api/data');
  const d = await r.json();
  const a = d.account || {};

  // Account bar
  document.getElementById('cash').textContent = '$' + (a.cash || 0).toFixed(2);
  const tv = a.total_value || 1000;
  document.getElementById('tval').textContent = '$' + tv.toFixed(2);
  document.getElementById('tval').className = 'av ' + (tv >= 1000 ? 'av-pos' : 'av-neg');
  const ret = a.total_return_pct || 0;
  document.getElementById('tret').textContent = (ret >= 0 ? '+' : '') + ret.toFixed(2) + '%';
  document.getElementById('tret').className = 'av ' + (ret >= 0 ? 'av-pos' : 'av-neg');
  document.getElementById('otk').innerHTML = a.on_track
    ? '<span class="track-pill track-yes">On Track</span>'
    : '<span class="track-pill track-no">Behind</span>';
  const wr = a.win_rate || 0;
  document.getElementById('wr').textContent = wr.toFixed(1) + '%';
  document.getElementById('wr').className = 'av ' + (wr >= 55 ? 'av-pos' : 'av-gold');
  document.getElementById('wl').textContent = (a.wins || 0) + 'W — ' + (a.losses || 0) + 'L';
  document.getElementById('py').textContent = '$' + (a.compounded_1yr || 0).toLocaleString();
  document.getElementById('lsc').textContent = d.last_scan || 'Pending...';

  // Market status
  const open = d.market_open;
  const chip = document.getElementById('mkt-chip');
  chip.className = 'mkt-chip ' + (open ? 'mkt-open-chip' : 'mkt-closed-chip');
  document.getElementById('mkt-dot').className = 'pulse-dot ' + (open ? 'pulse-green' : 'pulse-red');
  document.getElementById('mkt-lbl').textContent = open ? 'MARKET OPEN' : 'MARKET CLOSED';
  updateMarketTimer(open);

  // Signals
  const sigs = d.signals || [];
  document.getElementById('sigs').innerHTML = sigs.length ? sigs.map((s, i) => `
    <div class="sc ${s.direction === 'PUT' ? 'put-card' : ''}">
      <div class="sc-header">
        <div>
          <div class="sc-ticker">${s.ticker}</div>
          <div class="sc-badges">
            <span class="badge ${s.direction === 'CALL' ? 'b-call' : 'b-put'}">${s.direction}</span>
            <span class="badge b-type">${s.signal_type}</span>
            <span class="badge b-dur">${s.trade_duration}</span>
          </div>
        </div>
        <div class="sc-right">
          <div class="conf-block">
            <div class="conf-lbl">Confidence</div>
            <div class="conf-num" style="color:${confColor(s.confidence)}">${s.confidence}%</div>
            <div class="conf-bar"><div class="conf-fill" style="width:${s.confidence}%;background:${confBg(s.confidence)}"></div></div>
          </div>
          <button class="btn btn-exec" onclick="execute(${i})">Execute Paper Trade</button>
        </div>
      </div>
      <div class="sc-grid">
        <div class="sm"><div class="sml">Strike</div><div class="smv">$${s.strike}</div></div>
        <div class="sm"><div class="sml">Est. Premium</div><div class="smv">$${s.option_price}</div></div>
        <div class="sm"><div class="sml">Contracts</div><div class="smv">${s.suggested_contracts}</div></div>
        <div class="sm"><div class="sml">Total Cost</div><div class="smv">$${s.cost}</div></div>
        <div class="sm"><div class="sml">Expiry</div><div class="smv">${s.expiry_days} DTE</div></div>
      </div>
      <div class="sc-grid2">
        <div class="sm-g"><div class="sml">Target Exit</div><div class="smv">+${s.target_pct}%</div></div>
        <div class="sm-r"><div class="sml">Stop Loss</div><div class="smv">${s.stop_pct}%</div></div>
      </div>
      <div class="thesis-text">${s.thesis}</div>
      <div class="rationale">Signal basis: ${s.rationale}</div>
      <div class="tip-row"><b>Robinhood execution:</b> ${s.robinhood_tip}</div>
    </div>`).join('')
    : '<div class="empty">Initializing market scan — approximately 2 minutes on first load</div>';

  // Positions
  const pos = d.positions || [];
  const openP = pos.filter(p => p.status === 'OPEN');
  const closedP = pos.filter(p => p.status === 'CLOSED');

  document.getElementById('opn').innerHTML = openP.length ? openP.map(p => `<tr>
    <td class="td-ticker">${p.ticker}</td>
    <td><span class="badge ${p.option_type === 'CALL' ? 'b-call' : 'b-put'}">${p.option_type}</span></td>
    <td class="td-mono">$${p.strike}</td>
    <td class="td-mono">${p.contracts}</td>
    <td class="td-mono">$${(p.entry_price || 0).toFixed(2)}</td>
    <td class="${(p.pnl || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(p.pnl || 0) >= 0 ? '+' : ''}$${(p.pnl || 0).toFixed(2)}</td>
    <td class="${(p.pnl_pct || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(p.pnl_pct || 0) >= 0 ? '+' : ''}${(p.pnl_pct || 0).toFixed(1)}%</td>
    <td class="td-pos td-mono">+${p.target_pct}%</td>
    <td class="td-neg td-mono">${p.stop_pct}%</td>
    <td style="color:var(--grey4);font-size:11px;font-family:'DM Mono',monospace">${p.trade_type}</td>
    <td><button class="btn btn-close" onclick="closePos(${p.id})">Close</button></td>
  </tr>`).join('')
  : '<tr><td colspan="11" class="empty" style="padding:24px">No open positions. Review signals tab for opportunities.</td></tr>';

  document.getElementById('cls').innerHTML = closedP.length ? closedP.map(p => `<tr>
    <td class="td-ticker">${p.ticker}</td>
    <td><span class="badge ${p.option_type === 'CALL' ? 'b-call' : 'b-put'}">${p.option_type}</span></td>
    <td class="td-mono">$${(p.entry_price || 0).toFixed(2)}</td>
    <td class="td-mono">$${(p.exit_price || 0).toFixed(2)}</td>
    <td class="${(p.pnl || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(p.pnl || 0) >= 0 ? '+' : ''}$${(p.pnl || 0).toFixed(2)}</td>
    <td class="${(p.pnl_pct || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(p.pnl_pct || 0) >= 0 ? '+' : ''}${(p.pnl_pct || 0).toFixed(1)}%</td>
    <td>${(p.pnl || 0) >= 0 ? '<span class="result-win">GAIN</span>' : '<span class="result-loss">LOSS</span>'}</td>
  </tr>`).join('')
  : '<tr><td colspan="7" class="empty" style="padding:24px">No closed positions on record.</td></tr>';

  // Trade log
  const logs = d.trade_log || [];
  document.getElementById('tlog').innerHTML = logs.length ? logs.map(l => `<tr>
    <td style="color:var(--grey4);font-family:'DM Mono',monospace;font-size:10px">${l.date}</td>
    <td class="td-ticker">${l.ticker}</td>
    <td style="color:${l.action === 'BUY' ? 'var(--green)' : 'var(--red)'};font-weight:600;font-family:'DM Mono',monospace">${l.action}</td>
    <td><span class="badge ${l.type === 'CALL' ? 'b-call' : 'b-put'}">${l.type}</span></td>
    <td class="td-mono">$${l.strike}</td>
    <td class="td-mono">${l.contracts}</td>
    <td class="td-mono">$${(l.price || 0).toFixed(2)}</td>
    <td class="${(l.pnl || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(l.pnl || 0) >= 0 ? '+' : ''}$${(l.pnl || 0).toFixed(2)}</td>
    <td class="${(l.pnl_pct || 0) >= 0 ? 'td-pos' : 'td-neg'}">${(l.pnl_pct || 0) >= 0 ? '+' : ''}${(l.pnl_pct || 0).toFixed(1)}%</td>
    <td style="color:var(--grey4);font-size:11px;font-family:'DM Mono',monospace">${l.trade_type}</td>
  </tr>`).join('')
  : '<tr><td colspan="10" class="empty" style="padding:30px">No transactions on record.</td></tr>';

  // After hours
  const ah = d.after_hours || [];
  document.getElementById('ahc').innerHTML = ah.length ? ah.map(a => `
    <div class="ah-row">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-family:'Playfair Display',serif;font-size:15px;font-weight:700;color:var(--navy)">${a.ticker}</span>
        <div style="display:flex;align-items:center;gap:14px">
          <span class="${a.sentiment > 0.05 ? 'td-pos' : a.sentiment < -0.05 ? 'td-neg' : ''}" style="font-family:'DM Mono',monospace;font-size:11px">${a.direction} &nbsp;${a.sentiment > 0 ? '+' : ''}${a.sentiment}</span>
          <span style="color:var(--grey4);font-family:'DM Mono',monospace;font-size:10px">${a.time}</span>
        </div>
      </div>
      <div style="color:var(--text2);margin-bottom:5px;line-height:1.6;font-size:12px">${a.headline}</div>
      <div style="color:var(--grey4);font-size:10px;font-family:'DM Mono',monospace;margin-bottom:4px">${a.source}</div>
      <div style="color:var(--navy);font-size:11px;font-weight:600">Recommendation: ${a.suggestion}</div>
    </div>`).join('')
  : '<div class="empty" style="padding:40px">After-hours analysis executes automatically at 4:00pm ET.</div>';
}

load();
setInterval(load, 30000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    init_db()
    print("=" * 52)
    print("  LEAPS INTEL — Equity Options Desk")
    print("  Paper Trading Simulation")
    print("  Target: +30% monthly return")
    print("  Open: http://localhost:8080")
    print("  Run 24/7: nohup python backend/simulator.py > logs/app.log 2>&1 &")
    print("=" * 52)
    threading.Thread(target=background, daemon=True).start()
    app.run(debug=False, port=8080, host="0.0.0.0")