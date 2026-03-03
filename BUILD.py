import os, sys, subprocess

def run(cmd): subprocess.run(cmd, shell=True, check=False)
def mkdir(p): os.makedirs(p, exist_ok=True)
def write(path, content):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    open(path, "w").write(content)
    print(f"  ✓ {path}")

print("\n╔══════════════════════════════════════╗")
print("║  LEAPS::INTEL — Building project... ║")
print("╚══════════════════════════════════════╝\n")

print("► Creating folders...")
for d in ["ml","backend","frontend/src","data/historical","data/features","data/options","data/sentiment","models","reports","notebooks"]:
    mkdir(d); print(f"  ✓ {d}/")

print("\n► Installing packages (3-5 min)...")
pkgs = ["numpy pandas scipy matplotlib seaborn","scikit-learn xgboost lightgbm catboost joblib optuna","torch","yfinance pandas-datareader","ta pandas-ta","mibian backtesting","mlflow","vaderSentiment textblob nltk","anthropic","sqlalchemy","flask flask-cors flask-socketio requests beautifulsoup4 feedparser praw","schedule apscheduler plyer python-dotenv twilio","plotly dash","colorama tqdm pytz tabulate rich"]
for p in pkgs:
    print(f"  pip install {p}")
    run(f"pip install {p} -q")

print("\n► Writing files...")
write("ml/__init__.py","")
write("backend/__init__.py","")
write(".gitignore",".env\n__pycache__/\n*.pyc\nvenv/\ndata/historical/\nmodels/*.pkl\n.DS_Store\nmlruns/\n")
write(".env","# Fill these in\nANTHROPIC_API_KEY=your_key_here\nREDDIT_CLIENT_ID=your_id\nREDDIT_CLIENT_SECRET=your_secret\nREDDIT_USER_AGENT=leaps_intel/1.0\nALERT_EMAIL=your@gmail.com\nGMAIL_APP_PASSWORD=your_app_password\n")

print("\n╔══════════════════════════════════════╗")
print("║  Step 1 done! Now run:              ║")
print("║  python ml/download_data.py         ║")
print("╚══════════════════════════════════════╝")
