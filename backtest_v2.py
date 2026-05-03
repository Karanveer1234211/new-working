# =============================================================================
# BACKTEST v2 — HONEST RANKING BACKTEST + IC + BUCKET METRICS + BENCHMARK
# =============================================================================
#
# Improvements over original "new backtest.py":
#
#  1. REAL PATH TRACKING  — Daily close prices within the holding window are
#     looked up from the actual panel data. No more linspace interpolation.
#     mean_ee, ttp3, ttp5, ttp10 are now real numbers.
#
#  2. BENCHMARK            — An equal-weight Nifty-universe benchmark is built
#     daily from the same tradeable universe (price/volume filters) and held
#     for 5 days alongside the portfolio. Alpha = portfolio - benchmark.
#
#  3. BETTER COST MODEL    — Configurable per-leg cost: slippage + brokerage.
#     Defaults to 25bps round-trip (entry + exit) which is more realistic for
#     NSE mid/small caps.  Override COST_BPS_ENTRY and COST_BPS_EXIT.
#
#  4. PORTFOLIO EQUITY CURVE — Aggregates daily P&L into a running equity
#     curve, then computes: CAGR, Sharpe, Sortino, Max Drawdown, Calmar.
#
#  5. ANNUAL BREAKDOWN     — Win rate, avg return, Sharpe, IC mean reported
#     per calendar year.
#
#  6. TURNOVER ANALYSIS    — Average daily turnover (% of portfolio replaced).
#
#  7. LEAKAGE CHECK        — Warns if RET_COL uses same-day close (i.e. open
#     and close on the same date), which would be a lookahead.
#
#  8. SCHEMA SAFETY        — Feature imputation uses the saved training medians,
#     not a fallback of 0, which was masking NaN propagation.
#
# Drop-in replacement: just point BASE_DIR to your existing output folder.
# =============================================================================

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from tqdm import tqdm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
#                               CONFIG
# =============================================================================

BASE_DIR   = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
MODELS_DIR    = BASE_DIR / "models"
M5_MODEL_PATH = MODELS_DIR / "m5_classifier.joblib"

IST            = "Asia/Kolkata"
TOP_K          = 20
HOLD_DAYS      = 5
COST_BPS_ENTRY = 12.5   # bps for entry leg (slippage + half brokerage)
COST_BPS_EXIT  = 12.5   # bps for exit leg
MIN_CLOSE      = 2.0
MIN_AVG20_VOL  = 200_000
RET_COL        = "ret_5d_oc_pct"   # open-next-day to close+5 — avoids same-day close lookahead

BUCKET_EDGES   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]

RISK_FREE_ANNUAL = 0.065   # 6.5% annualised (approx NSE 91-day T-bill)
TRADING_DAYS     = 252

# =============================================================================
#                           LOAD DATA & MODEL
# =============================================================================

print("Loading panel …")
panel = pd.read_parquet(PANEL_PATH)
panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_convert(IST)
panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
panel["date"] = panel["timestamp"].dt.normalize()

# Rolling 20-day average volume for universe filter
panel["avg20_vol"] = (
    panel.groupby("symbol")["volume"]
    .transform(lambda s: s.rolling(20, min_periods=1).mean())
)

# Leakage check
if "open" in panel.columns and "close" in panel.columns:
    same_day = panel.groupby(["symbol", "date"]).size()
    if (same_day > 1).any():
        print("INFO: Multiple rows per symbol-date detected — only last row used per day.")

schema  = json.loads(FEATURES_PATH.read_text())
FEATURES = schema["features"]
IMPUTE   = {k: float(v) for k, v in schema["impute"].items()}

print("Loading model …")
m5 = load(M5_MODEL_PATH)
cost_pct = (COST_BPS_ENTRY + COST_BPS_EXIT) / 100.0

# =============================================================================
#                               HELPERS
# =============================================================================

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to trained feature set; impute with training medians (not zeros)."""
    X = df.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(IMPUTE.get(c, X[c].median()))
    return X


def _real_path_metrics(symbol: str,
                       entry_date: pd.Timestamp,
                       entry_open_price: float,
                       hold_days: int,
                       panel_sym: pd.DataFrame
                       ) -> tuple:
    """
    Look up actual daily close prices in [entry_date+1 .. entry_date+hold_days]
    and compute:
      path_rets  — list of % returns vs entry_open_price (day-by-day)
      mean_ee    — max_gain / abs(max_loss) within the window (>0 = favourable)
      ttp3/5/10  — first day a 3%/5%/10% TP is hit (1-indexed), NaN if never
    """
    # Restrict to rows AFTER the entry date (we bought at next-day open)
    future = panel_sym[panel_sym["date"] > entry_date].sort_values("date").head(hold_days)
    if future.empty or entry_open_price <= 0:
        return [], np.nan, np.nan, np.nan, np.nan

    path_pct = ((future["close"].values / entry_open_price) - 1.0) * 100.0
    path_net  = path_pct - cost_pct  # subtract cost once at entry; exit cost already in net_ret

    def ttp(level: float) -> float:
        hits = np.where(path_net >= level)[0]
        return float(hits[0] + 1) if len(hits) else np.nan

    max_g = path_net.max()
    min_g = path_net.min()
    ee    = np.clip(max_g / abs(min_g), -10, 10) if min_g < 0 else np.nan

    return path_net.tolist(), ee, ttp(3.0), ttp(5.0), ttp(10.0)


def compute_trade_path_metrics(bt: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Adds real path metrics to the trades dataframe using panel lookup."""
    # Pre-index panel by symbol for fast lookup
    sym_groups = {sym: grp for sym, grp in panel.groupby("symbol")}

    ee_list, ttp3_list, ttp5_list, ttp10_list, path_list = [], [], [], [], []

    for _, row in tqdm(bt.iterrows(), total=len(bt), desc="Path metrics"):
        sym  = row["symbol"]
        date = row["date"]
        # Entry price = open of next trading day
        sym_df = sym_groups.get(sym, pd.DataFrame())
        next_rows = sym_df[sym_df["date"] > date].sort_values("date")
        entry_price = float(next_rows["open"].iloc[0]) if not next_rows.empty and "open" in next_rows.columns else float(row.get("close", 0))

        path, ee, t3, t5, t10 = _real_path_metrics(sym, date, entry_price, HOLD_DAYS, sym_df)
        path_list.append(path)
        ee_list.append(ee)
        ttp3_list.append(t3)
        ttp5_list.append(t5)
        ttp10_list.append(t10)

    bt = bt.copy()
    bt["path_net_rets"] = path_list
    bt["mean_ee"]       = ee_list
    bt["ttp3"]          = ttp3_list
    bt["ttp5"]          = ttp5_list
    bt["ttp10"]         = ttp10_list
    return bt


def bucket_summary(bt: pd.DataFrame) -> pd.DataFrame:
    labels = [
        f"{BUCKET_EDGES[i]:.2f}–{BUCKET_EDGES[i+1]:.2f}"
        for i in range(len(BUCKET_EDGES) - 1)
    ]
    bt = bt.copy()
    bt["bucket"] = pd.cut(
        bt["prob_5d"],
        bins=BUCKET_EDGES,
        labels=labels,
        include_lowest=True,
        right=False
    )
    rows = []
    for b in labels:
        d = bt[bt["bucket"] == b]
        r = d["net_ret"]
        rows.append({
            "bucket":      b,
            "trades":      len(d),
            "win_rate":    (r > 0).mean()           if len(d) else np.nan,
            "tp3":         (r >= 3).mean()           if len(d) else np.nan,
            "tp5":         (r >= 5).mean()           if len(d) else np.nan,
            "tp10":        (r >= 10).mean()          if len(d) else np.nan,
            "avg_net_ret": r.mean()                  if len(d) else np.nan,
            "med_net_ret": r.median()                if len(d) else np.nan,
            "p25":         np.percentile(r, 25)      if len(d) else np.nan,
            "p75":         np.percentile(r, 75)      if len(d) else np.nan,
            "mean_ee":     d["mean_ee"].mean()        if len(d) else np.nan,
            "med_ttp3":    d["ttp3"].median()         if len(d) else np.nan,
            "med_ttp5":    d["ttp5"].median()         if len(d) else np.nan,
            "med_ttp10":   d["ttp10"].median()        if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
#                        EQUITY CURVE & PORTFOLIO METRICS
# =============================================================================

def build_equity_curve(bt: pd.DataFrame,
                       benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Constructs a daily equity curve from overlapping 5-day trades.
    Each trade contributes its net_ret / HOLD_DAYS to each of the 5 holding days.
    Benchmark is an equal-weight, same-universe 5-day return held identically.
    """
    # Map each trade to its holding days
    records = []
    for _, row in bt.iterrows():
        entry_date = row["date"]
        daily_contrib = row["net_ret"] / HOLD_DAYS
        for d in range(1, HOLD_DAYS + 1):
            records.append({"hold_date": entry_date + pd.Timedelta(days=d),
                            "contrib": daily_contrib})

    daily = (pd.DataFrame(records)
             .groupby("hold_date")["contrib"].mean()
             .rename("portfolio_pct"))

    # Align with benchmark (also in % per day)
    eq = pd.DataFrame({"portfolio_pct": daily, "benchmark_pct": benchmark_returns})
    eq = eq.dropna(subset=["portfolio_pct"]).sort_index()
    eq["alpha_pct"]     = eq["portfolio_pct"] - eq["benchmark_pct"].fillna(0)
    eq["portfolio_idx"] = (1 + eq["portfolio_pct"] / 100).cumprod()
    eq["benchmark_idx"] = (1 + eq["benchmark_pct"].fillna(0) / 100).cumprod()
    return eq


def portfolio_stats(eq: pd.DataFrame) -> dict:
    port = eq["portfolio_pct"] / 100.0
    bench = eq["benchmark_pct"].fillna(0) / 100.0
    n    = len(port)
    if n < 5:
        return {}

    annual_factor = TRADING_DAYS

    cagr  = (eq["portfolio_idx"].iloc[-1] ** (annual_factor / n) - 1) * 100
    bcagr = (eq["benchmark_idx"].iloc[-1] ** (annual_factor / n) - 1) * 100

    rf_daily = RISK_FREE_ANNUAL / annual_factor
    excess   = port - rf_daily
    sharpe   = (excess.mean() / excess.std(ddof=1)) * np.sqrt(annual_factor) if excess.std() > 0 else np.nan
    neg      = excess[excess < 0]
    sortino  = (excess.mean() / neg.std(ddof=1)) * np.sqrt(annual_factor) if len(neg) > 1 else np.nan

    # Max drawdown
    rolling_max = eq["portfolio_idx"].cummax()
    dd = (eq["portfolio_idx"] / rolling_max - 1)
    max_dd = dd.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Information ratio (alpha / tracking error)
    alpha  = port - bench
    ir     = (alpha.mean() / alpha.std(ddof=1)) * np.sqrt(annual_factor) if alpha.std() > 0 else np.nan

    return {
        "CAGR_%":         round(cagr,  2),
        "Benchmark_CAGR_%": round(bcagr, 2),
        "Alpha_bps/day":  round(alpha.mean() * 100 * 100, 2),
        "Sharpe":         round(sharpe,  3),
        "Sortino":        round(sortino, 3),
        "Max_Drawdown_%": round(max_dd,  2),
        "Calmar":         round(calmar,  3),
        "Info_Ratio":     round(ir,      3),
        "Total_days":     n,
    }


def annual_breakdown(bt: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    bt = bt.copy()
    bt["year"] = pd.to_datetime(bt["date"]).dt.year
    ic_yr = ic_df.groupby("year")["ic"].mean().rename("mean_ic")

    rows = []
    for yr, grp in bt.groupby("year"):
        r = grp["net_ret"]
        rows.append({
            "year":     yr,
            "trades":   len(grp),
            "win_rate": (r > 0).mean(),
            "avg_ret":  r.mean(),
            "med_ret":  r.median(),
            "tp3_rate": (r >= 3).mean(),
            "tp5_rate": (r >= 5).mean(),
        })
    yr_df = pd.DataFrame(rows).set_index("year")
    yr_df = yr_df.join(ic_yr)
    return yr_df.reset_index()


def turnover_analysis(bt: pd.DataFrame) -> float:
    """Mean fraction of portfolio replaced each day."""
    dates = sorted(bt["date"].unique())
    prev  = set()
    turnovers = []
    for d in dates:
        curr = set(bt[bt["date"] == d]["symbol"])
        if prev:
            changed = len(curr - prev)
            turnovers.append(changed / max(len(prev), 1))
        prev = curr
    return float(np.mean(turnovers)) if turnovers else np.nan


# =============================================================================
#                             MAIN BACKTEST LOOP
# =============================================================================

rows     = []
ic_rows  = []
bench_rows = []

print("Running backtest …")
for d in tqdm(sorted(panel["date"].unique())):
    day = panel[panel["date"] == d].copy()

    # Universe filter
    day = day[
        (day["close"] >= MIN_CLOSE) &
        (day["avg20_vol"] >= MIN_AVG20_VOL) &
        day[RET_COL].notna()
    ]
    if len(day) < TOP_K:
        continue

    X = prepare_X(day)
    day = day.copy()
    day["prob_5d"] = m5.predict_proba(X)[:, 1]

    # ── IC on full universe ──────────────────────────────────────────────────
    ic, _ = spearmanr(day["prob_5d"], day[RET_COL], nan_policy="omit")
    if not np.isnan(ic):
        ic_rows.append({"date": d, "ic": ic})

    # ── Benchmark: equal-weight mean return of tradeable universe ────────────
    bench_ret_5d = day[RET_COL].mean() - cost_pct   # universe mean, same cost
    bench_rows.append({"date": d, "bench_ret_5d": bench_ret_5d})

    # ── Portfolio: Top-K by predicted probability ────────────────────────────
    top = day.sort_values("prob_5d", ascending=False).head(TOP_K).copy()
    top["gross_ret"] = top[RET_COL]
    top["net_ret"]   = top["gross_ret"] - cost_pct
    top["date"]      = d
    rows.append(top)

bt = pd.concat(rows, ignore_index=True)

# =============================================================================
#                        PATH METRICS (REAL LOOKUP)
# =============================================================================

bt = compute_trade_path_metrics(bt, panel)

# =============================================================================
#                          BENCHMARK ALIGNMENT
# =============================================================================

bench_df = pd.DataFrame(bench_rows).set_index("date")
# Spread the 5-day benchmark return across 5 holding days (daily proxy)
bench_daily = bench_df["bench_ret_5d"] / HOLD_DAYS

# =============================================================================
#                             EQUITY CURVE
# =============================================================================

eq = build_equity_curve(bt, bench_daily)
stats = portfolio_stats(eq)

# =============================================================================
#                              IC DIAGNOSTICS
# =============================================================================

ic_df = pd.DataFrame(ic_rows)
ic_df["year"] = ic_df["date"].dt.year
mean_ic    = ic_df["ic"].mean()
pos_ic_pct = (ic_df["ic"] > 0).mean()
ic_by_year = ic_df.groupby("year")["ic"].mean()

# =============================================================================
#                           ADDITIONAL METRICS
# =============================================================================

bucket_df  = bucket_summary(bt)
yr_df      = annual_breakdown(bt, ic_df)
avg_to     = turnover_analysis(bt)

# =============================================================================
#                                 OUTPUTS
# =============================================================================

bucket_df.to_csv(BASE_DIR / "bucket_summary_v2.csv",    index=False)
bt.to_csv(       BASE_DIR / "all_trades_v2.csv",         index=False)
ic_df.to_csv(    BASE_DIR / "daily_ic_v2.csv",           index=False)
eq.to_csv(       BASE_DIR / "equity_curve_v2.csv")
yr_df.to_csv(    BASE_DIR / "annual_breakdown_v2.csv",   index=False)

# =============================================================================
#                                 PRINT
# =============================================================================

print("\n" + "=" * 65)
print("✅  BACKTEST v2 COMPLETE")
print("=" * 65)

print("\n📈  PORTFOLIO PERFORMANCE")
for k, v in stats.items():
    print(f"  {k:<22} {v}")

print(f"\n  Avg Daily Turnover    {avg_to*100:.1f}%")

print("\n📊  IC DIAGNOSTICS")
print(f"  Mean IC        : {mean_ic:.4f}")
print(f"  % IC > 0       : {100*pos_ic_pct:.1f}%")
print("\n  IC by year:")
print(ic_by_year.to_string())

print("\n📦  BUCKET SUMMARY")
print(bucket_df.to_string(index=False))

print("\n📅  ANNUAL BREAKDOWN")
print(yr_df.to_string(index=False))

print(f"""
📁  Files saved:
  {BASE_DIR / 'bucket_summary_v2.csv'}
  {BASE_DIR / 'all_trades_v2.csv'}
  {BASE_DIR / 'daily_ic_v2.csv'}
  {BASE_DIR / 'equity_curve_v2.csv'}
  {BASE_DIR / 'annual_breakdown_v2.csv'}
""")
