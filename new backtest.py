# ============================================================
# HONEST RANKING BACKTEST (NO 1D GATE) + IC + BUCKET METRICS
# ============================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from tqdm import tqdm
from scipy.stats import spearmanr

# ============================================================
# ========================== CONFIG ==========================
# ============================================================

BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")

PANEL_PATH    = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
MODELS_DIR    = BASE_DIR / "models"

M5_MODEL_PATH = MODELS_DIR / "m5_classifier.joblib"

IST = "Asia/Kolkata"

TOP_K = 20
HOLD_DAYS = 5
COST_BPS = 20.0

MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

RET_COL = "ret_5d_oc_pct"

BUCKET_EDGES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]

# ============================================================
# ====================== LOAD DATA ===========================
# ============================================================

print("Loading panel...")
panel = pd.read_parquet(PANEL_PATH)
panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_convert(IST)
panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
panel["date"] = panel["timestamp"].dt.normalize()

panel["avg20_vol"] = (
    panel.groupby("symbol")["volume"]
         .transform(lambda s: s.rolling(20, min_periods=1).mean())
)

schema = json.loads(FEATURES_PATH.read_text())
FEATURES = schema["features"]
IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

print("Loading model...")
m5 = load(M5_MODEL_PATH)

cost_pct = COST_BPS / 100.0

# ============================================================
# ====================== HELPERS =============================
# ============================================================

def prepare_X(df):
    X = df.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    return X


def compute_trade_path_metrics(df):
    def _calc(path):
        if not path or len(path) == 0:
            return np.nan, np.nan, np.nan
        path = np.asarray(path)

        max_ret = path.max()
        min_ret = path.min()

        ee = max_ret / abs(min_ret) if min_ret < 0 else np.nan
        ee = np.clip(ee, -10, 10)

        def ttp(level):
            hit = np.where(path >= level)[0]
            return int(hit[0] + 1) if len(hit) else np.nan

        return ee, ttp(3.0), ttp(5.0)

    vals = df["path_net_rets"].apply(_calc)
    df["mean_ee"], df["ttp3"], df["ttp5"] = zip(*vals)
    return df


def bucket_summary_full(df):
    labels = [
        f"{BUCKET_EDGES[i]:.2f}-{BUCKET_EDGES[i+1]:.2f}"
        for i in range(len(BUCKET_EDGES) - 1)
    ]

    df = df.copy()
    df["bucket"] = pd.cut(
        df["prob_5d"],
        bins=BUCKET_EDGES,
        labels=labels,
        include_lowest=True,
        right=False
    )

    rows = []

    for b in labels:
        d = df[df["bucket"] == b]
        if d.empty:
            rows.append({
                "bucket": b, "trades": 0,
                "win_rate": np.nan,
                "tp_hit_3pct": np.nan,
                "tp_hit_5pct": np.nan,
                "tp_hit_10pct": np.nan,
                "avg_net_ret": np.nan,
                "med_net_ret": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "mean_ee": np.nan,
                "median_ttp3": np.nan,
                "median_ttp5": np.nan,
            })
            continue

        r = d["net_ret"]

        rows.append({
            "bucket": b,
            "trades": len(d),
            "win_rate": (r > 0).mean(),
            "tp_hit_3pct": (r >= 3).mean(),
            "tp_hit_5pct": (r >= 5).mean(),
            "tp_hit_10pct": (r >= 10).mean(),
            "avg_net_ret": r.mean(),
            "med_net_ret": r.median(),
            "p25": np.percentile(r, 25),
            "p75": np.percentile(r, 75),
            "mean_ee": d["mean_ee"].mean(),
            "median_ttp3": d["ttp3"].median(),
            "median_ttp5": d["ttp5"].median(),
        })

    return pd.DataFrame(rows)


def compute_daily_ic(df):
    rows = []
    for d, g in df.groupby("date"):
        x = g["prob_5d"]
        y = g[RET_COL]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        ic, _ = spearmanr(x[mask], y[mask])
        rows.append({"date": d, "ic": ic})
    ic_df = pd.DataFrame(rows)
    ic_df["year"] = ic_df["date"].dt.year
    return ic_df

# ============================================================
# ===================== MAIN BACKTEST =========================
# ============================================================

rows = []
ic_rows = []

print("Running backtest...")

for d in tqdm(sorted(panel["date"].unique())):

    day = panel[panel["date"] == d].copy()

    # A: Universe filter only
    day = day[
        (day["close"] >= MIN_CLOSE) &
        (day["avg20_vol"] >= MIN_AVG20_VOL)
    ]
    if len(day) < 20:
        continue

    X = prepare_X(day)
    day["prob_5d"] = m5.predict_proba(X)[:, 1]

    # ---- IC (FULL UNIVERSE, NO TOP‑K) ----
    ic, _ = spearmanr(day["prob_5d"], day[RET_COL], nan_policy="omit")
    if ic == ic:
        ic_rows.append({"date": d, "ic": ic})

    # ---- Portfolio trades (TOP‑K) ----
    day = day.sort_values("prob_5d", ascending=False).head(TOP_K)

    day["gross_ret"] = day[RET_COL]
    day["net_ret"] = day["gross_ret"] - cost_pct

    day["path_net_rets"] = day["net_ret"].apply(
        lambda x: np.linspace(x / HOLD_DAYS, x, HOLD_DAYS).tolist()
    )

    rows.append(day)

bt = pd.concat(rows, ignore_index=True)

# ============================================================
# ====================== METRICS ==============================
# ============================================================

bt = compute_trade_path_metrics(bt)
bucket_df = bucket_summary_full(bt)

ic_df = pd.DataFrame(ic_rows)
ic_df["year"] = ic_df["date"].dt.year

mean_ic = ic_df["ic"].mean()
pos_ic_pct = (ic_df["ic"] > 0).mean()
ic_by_year = ic_df.groupby("year")["ic"].mean()

# ============================================================
# ======================== OUTPUT =============================
# ============================================================

bucket_df.to_csv(BASE_DIR / "bucket_summary_full_metrics.csv", index=False)
bt.to_csv(BASE_DIR / "all_trades_with_metrics.csv", index=False)
ic_df.to_csv(BASE_DIR / "daily_ic.csv", index=False)

print("\n✅ BACKTEST COMPLETE\n")

print("📊 IC DIAGNOSTICS")
print(f"Mean IC          : {mean_ic:.4f}")
print(f"% IC > 0         : {100*pos_ic_pct:.2f}%")
print("\nIC by year:")
print(ic_by_year)

print("\n📦 BUCKET SUMMARY")
print(bucket_df)
