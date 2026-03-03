# PersonalTradingBot

ML-powered options paper trading simulator targeting 30% monthly returns.

## What it does

- Scans 16 tickers every 5 minutes for day trade and swing trade setups
- Generates options signals with entry, target (+30-50%), and stop loss (-30-40%)
- Estimates realistic option premiums using Black-Scholes pricing
- Tracks a simulated $1,000 account with full P&L history
- Runs after-hours Reddit sentiment scan at 4pm ET daily
- Live countdown timer to market open/close
- Institutional UI — no emojis, clean professional design

## Stack

- **ML Model**: XGBoost + LightGBM + Random Forest ensemble (trained on 5yr historical data)
- **Features**: 50+ technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.)
- **Sentiment**: VADER NLP on Reddit (WSB, r/stocks, r/investing)
- **Backend**: Flask + SQLite
- **Frontend**: Vanilla JS — no framework needed
- **Data**: yfinance (free, real market data)

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run paper trader (opens at localhost:8080)
python backend/simulator.py

# Run 24/7 in background (no terminal needed)
nohup python backend/simulator.py > logs/app.log 2>&1 &

# Stop background process
pkill -f "python backend/simulator.py"

# Retrain ML model on fresh data
python ml/download_data.py
python ml/feature_engineering.py
python ml/train_model.py
python ml/backtest.py
```

## Signal Types

| Signal | Duration | Target | Stop |
|--------|----------|--------|------|
| Oversold Bounce | 1-2 DTE | +30% | -40% |
| Momentum Breakout | 3-5 days | +40% | -35% |
| Overbought Reversal | 2-4 days | +35% | -40% |
| Volatility Squeeze | 5-7 days | +50% | -30% |

## Disclaimer

This is an educational paper trading simulation. Not financial advice.
Options trading involves significant risk of loss. Never trade real money
without extensive paper trading experience first.
