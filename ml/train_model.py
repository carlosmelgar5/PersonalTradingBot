import pandas as pd, numpy as np, os, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from colorama import Fore, init
init(autoreset=True)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

FEATURES = ["rsi_14","rsi_7","macd","macd_signal","macd_hist","adx","adx_pos","adx_neg","cci",
    "bb_width","bb_pct","atr_pct","hv_20","hv_60","hv_90","mfi","vol_ratio","cmf",
    "returns_1d","returns_5d","returns_20d","returns_60d","gap","candle_body",
    "above_sma20","above_sma50","above_sma200","price_sma20_pct","price_sma50_pct",
    "52w_pct","stoch_k","stoch_d","roc_10","roc_30","williams","awesome",
    "golden_cross","death_cross","aroon_up","aroon_down"]

# Use 6-month target — balanced labels, enough historical data
TARGET = "target_6mo"

print(f"\n{Fore.YELLOW}Loading data...")
df = pd.read_csv("data/features/combined.csv", index_col=0, parse_dates=True)
df = df.dropna(subset=FEATURES + [TARGET])

# Only use data up to mid-2025 so test set has complete 6mo future
df = df[df.index <= "2025-06-01"]
print(f"  Rows after date filter: {len(df):,}")
print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"  Positive rate: {df[TARGET].mean()*100:.1f}%")

# 70/30 time split
split = df.index.sort_values()[int(len(df)*0.70)]
train, test = df[df.index <= split], df[df.index > split]
print(f"  Train: {len(train):,} ({train[TARGET].mean()*100:.1f}% positive)")
print(f"  Test:  {len(test):,} ({test[TARGET].mean()*100:.1f}% positive)")

X_tr, y_tr = train[FEATURES].values, train[TARGET].values
X_te, y_te = test[FEATURES].values,  test[TARGET].values

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
joblib.dump(scaler, "models/scaler.pkl")

spw = float((y_tr==0).sum()) / float((y_tr==1).sum() + 1e-9)

def show(name, model, X, y):
    yp  = model.predict(X)
    ypr = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, ypr)
    pre = precision_score(y, yp, zero_division=0)
    acc = accuracy_score(y, yp)
    f1  = f1_score(y, yp, zero_division=0)
    c   = Fore.GREEN if auc > 0.60 else Fore.YELLOW if auc > 0.55 else Fore.RED
    print(f"  {Fore.CYAN}{name}:  AUC={c}{auc:.3f}{Fore.RESET}  Prec={c}{pre*100:.1f}%{Fore.RESET}  Acc={acc*100:.1f}%")
    return dict(auc=round(auc,4), precision=round(pre,4), accuracy=round(acc,4), f1=round(f1,4))

results, fitted = {}, {}
configs = {
    "xgboost": xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10, gamma=0.3,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=spw,
        eval_metric="auc", random_state=42, n_jobs=-1),
    "lightgbm": lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
        num_leaves=15, min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6,
        min_samples_leaf=50, max_features=0.5,
        class_weight="balanced", random_state=42, n_jobs=-1),
}

for name, model in configs.items():
    print(f"\n{Fore.YELLOW}Training {name}...")
    model.fit(X_tr_s, y_tr)
    results[name] = show(name, model, X_te_s, y_te)
    joblib.dump(model, f"models/{name}.pkl")
    fitted[name] = model

print(f"\n{Fore.YELLOW}Training ensemble...")
ens = VotingClassifier(
    estimators=list(fitted.items()), voting="soft", weights=[3,3,2])
ens.fit(X_tr_s, y_tr)
results["ensemble"] = show("ensemble", ens, X_te_s, y_te)
joblib.dump(ens, "models/ensemble.pkl")

imp = pd.DataFrame({"feature":FEATURES,"importance":fitted["xgboost"].feature_importances_})
imp.sort_values("importance",ascending=False).to_csv("reports/feature_importance.csv",index=False)

print(f"\n{Fore.YELLOW}Top predictive features:")
for _, r in imp.sort_values("importance",ascending=False).head(8).iterrows():
    print(f"  {r.feature:<25} {'█'*int(r.importance*400)} {r.importance:.4f}")

print(f"\n{Fore.YELLOW}{'='*50}")
best = max(results, key=lambda k: results[k]["auc"])
for n,m in results.items():
    tag = " ◄ BEST" if n==best else ""
    print(f"  {n:<20} AUC={m['auc']:.3f}  Precision={m['precision']*100:.1f}%{tag}")
print(f"\n{Fore.GREEN}✓ Done! Run: python ml/backtest.py")
