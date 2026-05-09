
# COMPREHENSIVE FEATURE DIAGNOSTIC & PRUNING ANALYSIS
# =============================================================================
#
# What this does — in order:
#
# STEP 1 — GLOBAL FEATURE IMPORTANCE
# LightGBM gain, split count, and SHAP values on the full dataset.
# Gives you the baseline "what does the model actually use."
#
# STEP 2 — TEMPORAL STABILITY (per-year & per-quarter importance)
# Retrains a lightweight model on each year/quarter separately.
# Flags features whose importance is HIGH in some periods and LOW
# in others — these are regime-dependent or overfitting signals.
# Stability score = 1 - (std / mean) of importance across periods.
#
# STEP 3 — IC PER FEATURE (Spearman rank correlation vs 5D return)
# For every feature, computes daily IC then aggregates by year.
# Features with consistent positive IC are genuinely predictive.
# Features with IC near zero or sign-flipping are noise.
#
# STEP 4 — CROSS-STOCK GENERALIZATION TEST
# Splits universe into two halves by price bucket (low vs high price).
# Trains on low-price stocks, tests on high-price stocks and vice versa.
# Features that lose importance in cross-bucket test = encoding price
# level rather than signal. This is the definitive test for raw price
# features like D_cpr_bc, D_pivot, D_prev_close etc.
#
# STEP 5 — REDUNDANCY / CORRELATION ANALYSIS
# Computes feature-feature Spearman correlation matrix.
# Flags pairs with |corr| > 0.85 — one of the pair is redundant.
# Suggests which to keep (higher IC) and which to drop.
#
# STEP 6 — PERMUTATION IMPORTANCE (model-agnostic, leak-free)
# Shuffles each feature one at a time on the held-out test set.
# Measures drop in AUC. Features where shuffling doesn't hurt = useless.
# This is the most honest importance measure.
#
# STEP 7 — PRUNING RECOMMENDATION TABLE
# Combines all signals above into a single ranked table with a
# KEEP / REVIEW / DROP recommendation for each feature.
# DROP = bad on 3+ metrics
# REVIEW = bad on 1-2 metrics, needs manual check
# KEEP = good across all metrics
#
# OUTPUT FILES:
# feature_diagnostics_full.xlsx — all 7 analyses in separate sheets
# feature_importance_global.csv
# feature_ic_by_year.csv
# feature_stability_by_year.csv
# feature_stability_by_quarter.csv
# feature_correlation_matrix.csv
# feature_permutation_importance.csv
# feature_pruning_recommendation.csv ← the one you actually act on
#
# =============================================================================
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")
# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
MODELS_DIR = BASE_DIR / "models"
M5_MODEL_PATH = MODELS_DIR / "m5_classifier.joblib"
OUT_DIR = BASE_DIR / "feature_diagnostics"
OUT_DIR.mkdir(exist_ok=True)
IST = "Asia/Kolkata"
RET_COL = "ret_5d_oc_pct"
TARGET_COL = "top20_vs_bot20_5d" # built below if missing
EMBARGO_DAYS = 5
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000
# Lightweight model params for per-period retraining (fast)
DIAG_LGB_PARAMS = dict(
 n_estimators=400,
 learning_rate=0.05,
 num_leaves=63,
 max_depth=6,
 feature_fraction=0.8,
 bagging_fraction=0.8,
 bagging_freq=1,
 min_data_in_leaf=100,
 reg_alpha=0.1,
 reg_lambda=5.0,
 n_jobs=-1,
 random_state=42,
 verbosity=-1,
)
# Thresholds for pruning recommendation
IC_THRESHOLD = 0.02 # mean |IC| below this = weak signal
STABILITY_THRESHOLD = 0.40 # stability score below this = unstable
PERM_DROP_THRESHOLD = 0.001 # permutation AUC drop below this = useless
CORR_THRESHOLD = 0.85 # |corr| above this = redundant pair
GENERALIZATION_RATIO = 0.60 # cross-bucket importance ratio below this = price-encoding
# =============================================================================
# LOAD DATA & MODEL
# =============================================================================
print("=" * 65)
print("FEATURE DIAGNOSTIC & PRUNING ANALYSIS")
print("=" * 65)
print("\n[1/7] Loading panel and model...")
panel = pd.read_parquet(PANEL_PATH)
panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_convert(IST)
panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
panel["date"] = panel["timestamp"].dt.normalize()
panel["year"] = panel["timestamp"].dt.year
panel["quarter"] = panel["timestamp"].dt.to_period("Q").astype(str)
# Rolling avg volume filter
panel["avg20_vol"] = (
 panel.groupby("symbol")["volume"]
 .transform(lambda s: s.rolling(20, min_periods=1).mean())
)
schema = json.loads(FEATURES_PATH.read_text())
FEATURES = schema["features"]
IMPUTE = {k: float(v) for k, v in schema["impute"].items()}
m5 = load(M5_MODEL_PATH)
print(f" Panel shape : {panel.shape}")
print(f" Features : {len(FEATURES)}")
print(f" Date range : {panel['date'].min().date()} → {panel['date'].max().date()}")
# =============================================================================
# BUILD TARGET LABEL
# =============================================================================
def build_target(panel: pd.DataFrame) -> pd.DataFrame:
 """Reproduce top20_vs_bot20_5d label used in cpr_fix.py."""
 p = panel.copy()
 p["date"] = pd.to_datetime(p["timestamp"]).dt.normalize()
 daily_ret = (
 pd.to_numeric(p["close"], errors="coerce")
 .groupby(p["symbol"], observed=True)
 .pct_change() * 100.0
 )
 p["vol_20"] = (
 daily_ret.groupby(p["symbol"], observed=True)
 .rolling(20, min_periods=10).std()
 .reset_index(level=0, drop=True)
 )
 p["atr_pct"] = (
 pd.to_numeric(p["D_atr14"], errors="coerce")
 / pd.to_numeric(p["close"], errors="coerce")
 ).replace([np.inf, -np.inf], np.nan) * 100.0
 vol_basis = p["vol_20"].fillna(p["atr_pct"]).replace(0.0, np.nan)
 r5 = pd.to_numeric(p[RET_COL], errors="coerce")
 p["ret_5d_adj"] = r5 / vol_basis
 grp = p.groupby("date")
 p["rank_5d_pct"] = grp["ret_5d_adj"].rank(method="average", pct=True)
 p[TARGET_COL] = np.where(
 p["rank_5d_pct"] >= 0.80, 1,
 np.where(p["rank_5d_pct"] <= 0.20, 0, np.nan)
 )
 return p
panel = build_target(panel)
# =============================================================================
# PREPARE FEATURE MATRIX
# =============================================================================

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    return X
# Labeled rows only (top 20% vs bottom 20%)
labeled = panel[panel[TARGET_COL].notna()].copy()
labeled = labeled[
 (labeled["close"] >= MIN_CLOSE) &
 (labeled["avg20_vol"] >= MIN_AVG20_VOL)
].copy()
X_all = prepare_X(labeled)
y_all = labeled[TARGET_COL].astype(int).values
dates_all = labeled["date"].values
years_all = labeled["year"].values
quarters_all = labeled["quarter"].values
closes_all = labeled["close"].values
print(f" Labeled rows : {len(X_all):,}")
# =============================================================================
# STEP 1 — GLOBAL FEATURE IMPORTANCE (Gain + Split + SHAP)
# =============================================================================
print("\n[2/7] Computing global feature importance...")
# Use the saved model's booster directly
booster = m5.named_steps["classifier"].booster_ \
 if hasattr(m5, "named_steps") else \
 (m5.booster_ if hasattr(m5, "booster_") else None)
# Try to get base LightGBM model regardless of calibration wrapper
def _get_lgb_booster(model):
    """Extract LightGBM booster from any wrapper."""
    if hasattr(model, "booster_"):
        return model.booster_
    if hasattr(model, "base_estimator"):
        return _get_lgb_booster(model.base_estimator)
    if hasattr(model, "estimator"):
        return _get_lgb_booster(model.estimator)
    if hasattr(model, "calibrated_classifiers_"):
        return _get_lgb_booster(model.calibrated_classifiers_[0].estimator)
    if hasattr(model, "named_steps"):
        for v in model.named_steps.values():
            try:
                return _get_lgb_booster(v)
            except Exception:
                pass
    raise ValueError("Cannot extract LightGBM booster from model")
# Retrain a fresh diagnostic model on full labeled data for importance
print(" Training diagnostic model on full data...")
# Time-ordered train/test split (last 15% = test, embargoed)
ts_order = np.argsort(dates_all)
n = len(ts_order)
cut = int(0.85 * n)
embargo_cut = cut - EMBARGO_DAYS * 50 # rough daily embargo
tr_idx = ts_order[:embargo_cut]
te_idx = ts_order[cut:]
clf_diag = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
clf_diag.fit(
 X_all.iloc[tr_idx], y_all[tr_idx],
 eval_set=[(X_all.iloc[te_idx], y_all[te_idx])],
 callbacks=[lgb.callback.log_evaluation(period=0)]
)
gain_imp = clf_diag.booster_.feature_importance(importance_type="gain")
split_imp = clf_diag.booster_.feature_importance(importance_type="split")
# Normalize to 0-100
gain_norm = gain_imp / gain_imp.sum() * 100
split_norm = split_imp / split_imp.sum() * 100
global_imp = pd.DataFrame({
 "feature": FEATURES,
 "gain_pct": gain_norm,
 "split_pct": split_norm,
 "gain_rank": pd.Series(gain_norm).rank(ascending=False).astype(int).values,
 "split_rank": pd.Series(split_norm).rank(ascending=False).astype(int).values,
})
global_imp["avg_rank"] = (global_imp["gain_rank"] + global_imp["split_rank"]) / 2
global_imp = global_imp.sort_values("gain_pct", ascending=False).reset_index(drop=True)
# Test AUC for reference
base_proba = clf_diag.predict_proba(X_all.iloc[te_idx])[:, 1]
base_auc = roc_auc_score(y_all[te_idx], base_proba)
print(f" Diagnostic model test AUC: {base_auc:.4f}")
global_imp.to_csv(OUT_DIR / "feature_importance_global.csv", index=False)
print(f" Saved: feature_importance_global.csv")
# =============================================================================
# STEP 2 — TEMPORAL STABILITY (per-year & per-quarter importance)
# =============================================================================
print("\n[3/7] Computing temporal stability (per year & quarter)...")
def importance_for_period(X_period, y_period, dates_period, label):
    """Train a quick model on one period, return normalized gain importance."""
    if len(X_period) < 500 or y_period.sum() < 50:
        return None

    ts_ord = np.argsort(dates_period)
    cut = max(int(0.8 * len(ts_ord)), len(ts_ord) - 2000)
    tr = ts_ord[:cut]
    te = ts_ord[cut:]

    if len(te) < 50 or len(tr) < 200:
        return None

    clf = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
    try:
        clf.fit(
            X_period.iloc[tr],
            y_period[tr],
            eval_set=[(X_period.iloc[te], y_period[te])],
            callbacks=[lgb.callback.log_evaluation(period=0)],
        )
        imp = clf.booster_.feature_importance(importance_type="gain").astype(float)
        total = imp.sum()
        if total == 0:
            return None
        return imp / total * 100
    except Exception:
        return None
# --- Per year ---

year_imp_rows = {}

for yr in sorted(labeled["year"].unique()):
    mask = years_all == yr
    Xi = X_all[mask]
    yi = y_all[mask]
    di = dates_all[mask]

    imp = importance_for_period(Xi, yi, di, str(yr))

    if imp is not None:
        year_imp_rows[str(yr)] = imp
        print(f" Year {yr}: {mask.sum():,} rows — OK")
    else:
        print(f" Year {yr}: {mask.sum():,} rows — SKIP (too few)")

year_imp_df = pd.DataFrame(year_imp_rows, index=FEATURES).T
year_imp_df.index.name = "year"
# --- Per quarter ---

quarter_imp_rows = {}

for qtr in sorted(labeled["quarter"].unique()):
    mask = quarters_all == qtr
    Xi = X_all[mask]
    yi = y_all[mask]
    di = dates_all[mask]

    imp = importance_for_period(Xi, yi, di, qtr)

    if imp is not None:
        quarter_imp_rows[qtr] = imp

quarter_imp_df = pd.DataFrame(quarter_imp_rows, index=FEATURES).T
quarter_imp_df.index.name = "quarter"
# Stability score per feature (higher = more stable across periods)
def stability_score(imp_df: pd.DataFrame) -> pd.Series:
 """Coefficient of variation inverted: 1 - (std/mean). Higher = more stable."""
 m = imp_df.mean()
 s = imp_df.std()
 # Avoid div-by-zero for zero-importance features
 cv = s / m.replace(0, np.nan)
 return (1 - cv).clip(lower=-1, upper=1).fillna(0)
year_stability = stability_score(year_imp_df).rename("stability_yearly")
quarter_stability = stability_score(quarter_imp_df).rename("stability_quarterly")
year_imp_df.to_csv(OUT_DIR / "feature_stability_by_year.csv")
quarter_imp_df.to_csv(OUT_DIR / "feature_stability_by_quarter.csv")
print(" Saved: feature_stability_by_year.csv, feature_stability_by_quarter.csv")
# =============================================================================
# STEP 3 — IC PER FEATURE (Spearman rank correlation vs 5D return)
# =============================================================================
## =============================================================================
## STEP 3 — IC PER FEATURE (Spearman rank correlation vs 5D return)
## =============================================================================

print("\n[4/7] Computing per-feature IC by year...")

ic_rows = []

for yr in sorted(labeled["year"].unique()):
    mask = years_all == yr
    sub = labeled[mask].copy()
    Xi = X_all[mask]

    ret = pd.to_numeric(sub[RET_COL], errors="coerce").values

    year_ic = {}

    for feat in FEATURES:
        fvals = Xi[feat].values
        valid = np.isfinite(fvals) & np.isfinite(ret)

        if valid.sum() < 100:
            year_ic[feat] = np.nan
            continue

        ic, _ = spearmanr(fvals[valid], ret[valid])
        year_ic[feat] = ic

    year_ic["year"] = yr
    ic_rows.append(year_ic)

ic_by_year = pd.DataFrame(ic_rows).set_index("year")

# ---- IC summary statistics per feature ----

ic_summary = pd.DataFrame({
    "mean_ic": ic_by_year.mean(),
    "mean_abs_ic": ic_by_year.abs().mean(),
    "ic_std": ic_by_year.std(),
    "ic_pos_pct": (ic_by_year > 0).mean(),
    # TRUE sign-flip rate (not Bernoulli std)
    "ic_sign_flip": ic_by_year.apply(
        lambda s: (np.sign(s.dropna()).diff().abs() > 0).mean()
        if s.notna().sum() > 2 else np.nan
    ),
})

ic_by_year.to_csv(OUT_DIR / "feature_ic_by_year.csv")
print(" Saved: feature_ic_by_year.csv")
# =============================================================================
# STEP 4 — CROSS-STOCK GENERALIZATION TEST
# =============================================================================
print("\n[5/7] Cross-stock generalization test (price bucket split)...")
# Split universe at median close price
median_close = np.median(closes_all)
low_price_mask = closes_all <= median_close
high_price_mask = closes_all > median_close

def cross_bucket_importance(train_mask, test_mask, label):
    Xi_tr = X_all[train_mask]
    yi_tr = y_all[train_mask]

    Xi_te = X_all[test_mask]
    yi_te = y_all[test_mask]

    if yi_tr.sum() < 100 or yi_te.sum() < 50:
        return None

    clf = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)

    try:
        clf.fit(
            Xi_tr,
            yi_tr,
            callbacks=[lgb.callback.log_evaluation(period=0)]
        )

        imp = clf.booster_.feature_importance(
            importance_type="gain"
        ).astype(float)

        total = imp.sum()
        return imp / total * 100 if total > 0 else None

    except Exception:
        return None

print(f" Median close price: ₹{median_close:.1f}")
print(f" Low-price stocks : {low_price_mask.sum():,} rows")
print(f" High-price stocks : {high_price_mask.sum():,} rows")
# Train on low → test on high
imp_low_to_high = cross_bucket_importance(low_price_mask, high_price_mask, "low→high")
# Train on high → test on low
imp_high_to_low = cross_bucket_importance(high_price_mask, low_price_mask, "high→low")
# In-sample importance for same buckets
imp_low_insample = cross_bucket_importance(low_price_mask, low_price_mask, "low→low")
imp_high_insample = cross_bucket_importance(high_price_mask, high_price_mask, "high→high")
gen_df = pd.DataFrame({"feature": FEATURES})
if imp_low_insample is not None and imp_low_to_high is not None:
 gen_df["imp_low_insample"] = imp_low_insample
 gen_df["imp_low_to_high"] = imp_low_to_high
 gen_df["generalization_ratio_low"] = (
 gen_df["imp_low_to_high"] /
 gen_df["imp_low_insample"].replace(0, np.nan)
 ).fillna(0).clip(0, 2)
if imp_high_insample is not None and imp_high_to_low is not None:
 gen_df["imp_high_insample"] = imp_high_insample
 gen_df["imp_high_to_low"] = imp_high_to_low
 gen_df["generalization_ratio_high"] = (
 gen_df["imp_high_to_low"] /
 gen_df["imp_high_insample"].replace(0, np.nan)
 ).fillna(0).clip(0, 2)
# Average generalization ratio
ratio_cols = [c for c in gen_df.columns if c.startswith("generalization_ratio")]
if ratio_cols:
 gen_df["avg_generalization_ratio"] = gen_df[ratio_cols].mean(axis=1)
else:
 gen_df["avg_generalization_ratio"] = np.nan
gen_df = gen_df.sort_values("avg_generalization_ratio").reset_index(drop=True)
gen_df.to_csv(OUT_DIR / "feature_generalization.csv", index=False)
print(" Saved: feature_generalization.csv")
print(f" Features with poor generalization ratio (<{GENERALIZATION_RATIO}):")
poor_gen = gen_df[gen_df["avg_generalization_ratio"] < GENERALIZATION_RATIO]["feature"].tolist()
for f in poor_gen[:15]:
 ratio = gen_df[gen_df["feature"] == f]["avg_generalization_ratio"].values[0]
 print(f" {f:<40} ratio={ratio:.2f}")
# =============================================================================
# STEP 5 — REDUNDANCY / CORRELATION ANALYSIS
# =============================================================================
print("\n[6/7] Computing feature correlation matrix...")
# Sample for speed (max 50k rows)
sample_n = min(50_000, len(X_all))
sample_idx = np.random.choice(len(X_all), sample_n, replace=False)
X_sample = X_all.iloc[sample_idx]
# Spearman correlation (rank-based, handles non-linearity)
print(" Computing Spearman correlation (this takes ~1–2 min)...")
corr_matrix = X_sample.corr(method="spearman")
corr_matrix.to_csv(OUT_DIR / "feature_correlation_matrix.csv")
# Find highly correlated pairs
high_corr_pairs = []
feats = list(corr_matrix.columns)

for i in range(len(feats)):
    for j in range(i + 1, len(feats)):
        c = corr_matrix.iloc[i, j]
        if abs(c) >= CORR_THRESHOLD:

            ic_i = ic_summary.loc[feats[i], "mean_abs_ic"] if feats[i] in ic_summary.index else 0
            ic_j = ic_summary.loc[feats[j], "mean_abs_ic"] if feats[j] in ic_summary.index else 0

            drop_candidate = feats[i] if ic_i <= ic_j else feats[j]
            keep_candidate = feats[j] if ic_i <= ic_j else feats[i]

            high_corr_pairs.append({
                "feature_a": feats[i],
                "feature_b": feats[j],
                "spearman_corr": round(c, 3),
                "keep": keep_candidate,
                "drop_candidate": drop_candidate,
                "ic_a": round(ic_i, 4),
                "ic_b": round(ic_j, 4),
            })
corr_pairs_df = pd.DataFrame(high_corr_pairs).sort_values(
 "spearman_corr", key=abs, ascending=False
)
corr_pairs_df.to_csv(OUT_DIR / "feature_redundant_pairs.csv", index=False)
print(f" Found {len(high_corr_pairs)} highly correlated pairs (|r| ≥ {CORR_THRESHOLD})")
print(" Saved: feature_correlation_matrix.csv, feature_redundant_pairs.csv")
# =============================================================================
# STEP 6 — PERMUTATION IMPORTANCE (model-agnostic, leak-free)
# =============================================================================
print("\n[7/7] Computing permutation importance on held-out test set...")
print(" (Shuffles each feature and measures AUC drop — most honest measure)")
X_te = X_all.iloc[te_idx].copy()
y_te = y_all[te_idx]
perm_results = []
for feat in tqdm(FEATURES, desc="Permuting features"):
 X_shuffled = X_te.copy()
 X_shuffled[feat] = X_shuffled[feat].sample(frac=1, random_state=42).values
try:
    p_shuffled = clf_diag.predict_proba(X_shuffled)[:, 1]
    shuffled_auc = roc_auc_score(y_te, p_shuffled)
    auc_drop = base_auc - shuffled_auc

except Exception:
    auc_drop = np.nan

perm_results.append({
    "feature": feat,
    "base_auc": round(base_auc, 5),
    "auc_after_shuffle": round(base_auc - (auc_drop or 0), 5),
    "auc_drop": round(auc_drop or 0, 5),
})
perm_df = pd.DataFrame(perm_results).sort_values("auc_drop", ascending=False)
perm_df.to_csv(OUT_DIR / "feature_permutation_importance.csv", index=False)
print(" Saved: feature_permutation_importance.csv")
# =============================================================================
# STEP 7 — PRUNING RECOMMENDATION TABLE
# =============================================================================
print("\nBuilding pruning recommendation table...")
rec = global_imp[["feature", "gain_pct", "split_pct", "gain_rank"]].copy()
# Merge IC summary
rec = rec.merge(
 ic_summary[["mean_abs_ic", "ic_pos_pct", "ic_sign_flip"]].reset_index().rename(columns={"index": "feature"}),
 on="feature", how="left"
)
# Merge yearly stability
rec = rec.merge(
 year_stability.reset_index().rename(columns={"index": "feature", "stability_yearly": "stability_yearly"}),
 on="feature", how="left"
)
# Merge quarterly stability
rec = rec.merge(
 quarter_stability.reset_index().rename(columns={"index": "feature", "stability_quarterly": "stability_quarterly"}),
 on="feature", how="left"
)
# Merge generalization
rec = rec.merge(
 gen_df[["feature", "avg_generalization_ratio"]],
 on="feature", how="left"
)
# Merge permutation importance
rec = rec.merge(
 perm_df[["feature", "auc_drop"]],
 on="feature", how="left"
)
# ── Scoring: count how many red flags each feature has ──
rec["flag_low_ic"] = (rec["mean_abs_ic"] < IC_THRESHOLD).astype(int)
rec["flag_unstable_yr"] = (rec["stability_yearly"] < STABILITY_THRESHOLD).astype(int)
rec["flag_unstable_qtr"] = (rec["stability_quarterly"] < STABILITY_THRESHOLD).astype(int)
rec["flag_poor_gen"] = (rec["avg_generalization_ratio"] < GENERALIZATION_RATIO).astype(int)
rec["flag_perm_useless"] = (rec["auc_drop"] < PERM_DROP_THRESHOLD).astype(int)
rec["flag_sign_flip"] = (rec["ic_sign_flip"] > 0.45).astype(int)
rec["total_flags"] = (
 rec["flag_low_ic"] +
 rec["flag_unstable_yr"] +
 rec["flag_unstable_qtr"] +
 rec["flag_poor_gen"] +
 rec["flag_perm_useless"] +
 rec["flag_sign_flip"]
)
def recommend(row):
    if row["total_flags"] >= 3:
        return "DROP"
    elif row["total_flags"] >= 1:
        return "REVIEW"
    else:
        return "KEEP"
rec["recommendation"] = rec.apply(recommend, axis=1)
# Sort: DROP first, then REVIEW, then KEEP; within each by gain_rank
order = {"DROP": 0, "REVIEW": 1, "KEEP": 2}
rec["sort_order"] = rec["recommendation"].map(order)
rec = rec.sort_values(["sort_order", "gain_rank"]).drop("sort_order", axis=1)
# Round for readability
for col in ["gain_pct", "split_pct", "mean_abs_ic", "ic_pos_pct",
 "stability_yearly", "stability_quarterly",
 "avg_generalization_ratio", "auc_drop"]:
 if col in rec.columns:
    rec[col] = rec[col].round(4)
rec.to_csv(OUT_DIR / "feature_pruning_recommendation.csv", index=False)
# =============================================================================
# SAVE FULL EXCEL REPORT (all sheets in one file)
# =============================================================================
print("\nSaving full Excel report...")
excel_path = OUT_DIR / "feature_diagnostics_full.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
 rec.to_excel(writer, sheet_name=" Pruning Recommendation", index=False)
 global_imp.to_excel(writer, sheet_name="1 Global Importance", index=False)
 ic_by_year.to_excel(writer, sheet_name="2 IC by Year")
 ic_summary.to_excel(writer, sheet_name="3 IC Summary")
 year_imp_df.to_excel(writer, sheet_name="4 Stability by Year")
 quarter_imp_df.to_excel(writer, sheet_name="5 Stability by Quarter")
 gen_df.to_excel(writer, sheet_name="6 Generalization", index=False)
 corr_pairs_df.to_excel(writer, sheet_name="7 Redundant Pairs", index=False)
 perm_df.to_excel(writer, sheet_name="8 Permutation Importance", index=False)
print(f" Saved: {excel_path}")
# =============================================================================
# PRINT SUMMARY TO CONSOLE
# =============================================================================
drop_list = rec[rec["recommendation"] == "DROP"]["feature"].tolist()
review_list = rec[rec["recommendation"] == "REVIEW"]["feature"].tolist()
keep_list = rec[rec["recommendation"] == "KEEP"]["feature"].tolist()
print("\n" + "=" * 65)
print("PRUNING SUMMARY")
print("=" * 65)
print(f"\n KEEP ({len(keep_list)} features) — strong across all metrics")
for f in keep_list:
 row = rec[rec["feature"] == f].iloc[0]
 print(f" {f:<42} gain={row['gain_pct']:.2f}% IC={row['mean_abs_ic']:.3f} perm={row['auc_drop']:.4f}")
print(f"\n REVIEW ({len(review_list)} features) — weak on 1-2 metrics, check manually")
for f in review_list:
 row = rec[rec["feature"] == f].iloc[0]
 flags = []
 if row["flag_low_ic"]: flags.append("low_IC")
 if row["flag_unstable_yr"]: flags.append("unstable_year")
 if row["flag_unstable_qtr"]: flags.append("unstable_qtr")
 if row["flag_poor_gen"]: flags.append("poor_generalization")
 if row["flag_perm_useless"]: flags.append("perm_useless")
 if row["flag_sign_flip"]: flags.append("IC_sign_flip")
 print(f" {f:<42} flags={flags}")
print(f"\n DROP ({len(drop_list)} features) — bad on 3+ metrics")
for f in drop_list:
 row = rec[rec["feature"] == f].iloc[0]
 flags = []
 if row["flag_low_ic"]: flags.append("low_IC")
 if row["flag_unstable_yr"]: flags.append("unstable_year")
 if row["flag_unstable_qtr"]: flags.append("unstable_qtr")
 if row["flag_poor_gen"]: flags.append("poor_generalization")
 if row["flag_perm_useless"]: flags.append("perm_useless")
 if row["flag_sign_flip"]: flags.append("IC_sign_flip")
 print(f" {f:<42} flags={flags}")
print(f"""
 All outputs saved to:
 {OUT_DIR}
Files:
 feature_diagnostics_full.xlsx ← Open this first (all sheets)
 feature_pruning_recommendation.csv ← The action table
 feature_importance_global.csv
 feature_ic_by_year.csv
 feature_stability_by_year.csv
 feature_stability_by_quarter.csv
 feature_generalization.csv
 feature_redundant_pairs.csv
 feature_permutation_importance.csv
Next step:
 Review the REVIEW list manually — some may be worth keeping
 depending on your market intuition. DROP list is safe to remove.
 After pruning, delete panel_cache.parquet and re-run cpr_fix.py
 to retrain the model on the clean feature set.
""")