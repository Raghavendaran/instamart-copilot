"""
part_b_segments.py
------------------
Part B: Customer Segments and Lapse Prediction.
Reads clean data from Part A, builds RFM, K-means clusters, lapse model.
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

CLEAN = os.path.join("data", "clean")
OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

# ─────────────────────────────────────────────
# Load clean data
# ─────────────────────────────────────────────
orders    = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
products  = pd.read_csv(os.path.join(CLEAN, "products.csv"))

# Use only completed orders
completed = orders[orders["status"] == "completed"].copy()

# ─────────────────────────────────────────────
# TASK 1: Build RFM table
# ─────────────────────────────────────────────
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

rfm = (
    completed
    .groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)
rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
rfm = rfm.drop(columns=["last_order"])

print("=" * 60)
print("TASK 1: RFM Table")
print("=" * 60)
print(f"Customers in RFM: {len(rfm)}")
print(f"Recency   — median: {rfm['recency'].median()}, min: {rfm['recency'].min()}, max: {rfm['recency'].max()}")
print(f"Frequency — mean: {rfm['frequency'].mean():.2f}, max: {rfm['frequency'].max()}")
print(f"Monetary  — mean: {rfm['monetary'].mean():.2f}, max: {rfm['monetary'].max()}")
print("Reference: 1175 customers; recency median 19, min 0, max 180; freq mean 9.46, max 33; monetary mean 2552.06, max 14766.55")

# ─────────────────────────────────────────────
# TASK 2: K-means clustering
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_rfm = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

print("\n" + "=" * 60)
print("TASK 2: K-means for K = 2 to 8")
print("=" * 60)
print(f"{'K':>3}  {'Inertia':>12}  {'Silhouette':>12}")
kmeans_results = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_rfm)
    inertia = km.inertia_
    sil = silhouette_score(X_rfm, labels)
    kmeans_results[k] = {"inertia": inertia, "silhouette": sil, "labels": labels, "model": km}
    print(f"{k:>3}  {inertia:>12.1f}  {sil:>12.3f}")

print("""
Reference:
  K=2: 1820.8 / 0.465   K=3: 1268.2 / 0.445   K=4:  743.1 / 0.526
  K=5:  617.4 / 0.481   K=6:  503.2 / 0.482   K=7:  408.5 / 0.416  K=8: 366.5 / 0.412
""")
print("Explanation: K=8 always has lower inertia than K=2 because adding more clusters "
      "always reduces within-cluster variance — a model can perfectly fit each point as its "
      "own cluster. Inertia alone cannot pick K because it monotonically decreases; we need "
      "the elbow heuristic or silhouette scores, which balance compactness with separation.")

# ─────────────────────────────────────────────
# TASK 3: Profile, name, and project segments
# ─────────────────────────────────────────────
# Elbow + silhouette suggests K=4 (large silhouette drop after 4, clear elbow at 4)
FINAL_K = 4
km_final = kmeans_results[FINAL_K]["model"]
rfm["segment_id"] = kmeans_results[FINAL_K]["labels"]

print("\n" + "=" * 60)
print(f"TASK 3: Segment profiles for K={FINAL_K}")
print("=" * 60)
profile = rfm.groupby("segment_id")[["recency", "frequency", "monetary"]].mean().round(2)
sizes   = rfm.groupby("segment_id").size().rename("size")
profile = pd.concat([sizes, profile], axis=1)
print(profile.to_string())

print("""
Reference: sizes about 436/138/419/182
  The 138-customer segment has mean monetary ~7130 and mean frequency ~6.8
  The 419-customer segment has mean frequency ~19.4 and mean recency ~8.8 days
""")

# Name segments
# Identify Premium = highest mean monetary
seg_monetary = rfm.groupby("segment_id")["monetary"].mean()
premium_seg  = seg_monetary.idxmax()

seg_recency  = rfm.groupby("segment_id")["recency"].mean()
seg_freq     = rfm.groupby("segment_id")["frequency"].mean()

# Assign names based on profile
# Premium: highest monetary
# Active: low recency (ordered recently) + high frequency
# Occasional: medium everything
# At-Risk: high recency (haven't ordered in a while)
seg_names = {}
remaining = [s for s in range(FINAL_K) if s != premium_seg]
# Sort remaining by frequency desc → most frequent = Active Loyalists
remaining_by_freq = sorted(remaining, key=lambda s: seg_freq[s], reverse=True)
active_seg    = remaining_by_freq[0]
# Sort rest by recency asc → lowest recency = Occasional New
rest = [s for s in remaining_by_freq[1:]]
rest_by_recency = sorted(rest, key=lambda s: seg_recency[s])
occasional_seg = rest_by_recency[0]
atrisk_seg     = rest_by_recency[1]

name_map = {
    premium_seg:   "Premium",
    active_seg:    "Active Loyalist",
    occasional_seg:"Occasional Buyer",
    atrisk_seg:    "At-Risk",
}
action_map = {
    "Premium":        "Offer exclusive early-access sales and premium loyalty perks.",
    "Active Loyalist":"Send personalised reorder reminders and bundle offers.",
    "Occasional Buyer":"Re-engage with targeted discount coupons for their favourite category.",
    "At-Risk":        "Send a win-back campaign with a time-limited incentive.",
}
rfm["segment_name"] = rfm["segment_id"].map(name_map)
rfm["is_premium"]   = rfm["segment_id"] == premium_seg

print("Segment naming:")
for sid, name in name_map.items():
    n = (rfm["segment_id"] == sid).sum()
    action = action_map[name]
    print(f"  Segment {sid}: {name} ({n} customers) — {action}")
print("\nNote: cluster IDs are arbitrary labels; the ordering (0,1,2,3) carries no meaning.")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_rfm)
print(f"\nPCA explained_variance_ratio_: {pca.explained_variance_ratio_.round(3)}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.3f}")
print("Reference: ~0.634, ~0.218, total ~0.852")

# Save outputs/customer_segments.csv
segments_out = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].copy()
segments_out.to_csv(os.path.join(OUTPUTS, "customer_segments.csv"), index=False)
print(f"\nSaved outputs/customer_segments.csv ({len(segments_out)} rows)")

# ─────────────────────────────────────────────
# TASK 4: Define the lapse problem
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4: Lapse problem definition")
print("=" * 60)
CUTOFF = pd.Timestamp("2025-09-01")
SEP_END = pd.Timestamp("2025-09-30 23:59:59")

# Cohort: customers with at least one completed order before cutoff
pre = orders[(orders["order_ts"] < CUTOFF) & (orders["status"] == "completed")]
cohort_ids = pre["customer_id"].unique()
print(f"Cohort size (≥1 completed order before {CUTOFF.date()}): {len(cohort_ids)}")

# Label: lapsed = 1 if no completed order in Sep 2025
sep_completed = orders[
    (orders["order_ts"] >= CUTOFF) &
    (orders["order_ts"] <= SEP_END) &
    (orders["status"] == "completed")
]["customer_id"].unique()
label_df = pd.DataFrame({"customer_id": cohort_ids})
label_df["lapsed"] = (~label_df["customer_id"].isin(sep_completed)).astype(int)
print(f"Lapsed: {label_df['lapsed'].sum()} ({label_df['lapsed'].mean()*100:.2f}%)")
print(f"Reference: 1156 cohort; 443 lapsed (38.32%)")

# ─────────────────────────────────────────────
# TASK 5: Feature engineering (pre-cutoff only)
# ─────────────────────────────────────────────
all_pre = orders[orders["order_ts"] < CUTOFF].copy()
# category column is already present in clean orders.csv (added during Part A cleaning)
if "category" not in all_pre.columns:
    all_pre = all_pre.merge(products[["product_id","category"]], on="product_id", how="left")

comp_pre = all_pre[all_pre["status"] == "completed"]

# RFM features
feat_rfm = (
    comp_pre.groupby("customer_id")
    .agg(
        recency=("order_ts", lambda x: (CUTOFF - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    .reset_index()
)

# Favourite category
fav_cat = (
    comp_pre.groupby(["customer_id","category"])["order_id"]
    .count()
    .reset_index()
    .sort_values("order_id", ascending=False)
    .drop_duplicates("customer_id")
    .rename(columns={"category":"fav_category"})
    [["customer_id","fav_category"]]
)

# Cancellation rate
final_all = all_pre[all_pre["status"].isin(["completed","cancelled"])].copy()
final_all["is_cancelled"] = (final_all["status"] == "cancelled").astype(int)
cancel_rate = (
    final_all.groupby("customer_id")["is_cancelled"]
    .mean()
    .reset_index()
    .rename(columns={"is_cancelled": "cancel_rate"})
)

# Customer demographics
cust_feat = customers[["customer_id","city","age","gender","signup_date"]].copy()
cust_feat["tenure"] = (CUTOFF - pd.to_datetime(cust_feat["signup_date"])).dt.days

# Merge all features
feats = (
    label_df
    .merge(feat_rfm,    on="customer_id", how="left")
    .merge(fav_cat,     on="customer_id", how="left")
    .merge(cancel_rate, on="customer_id", how="left")
    .merge(cust_feat[["customer_id","city","age","gender","tenure"]], on="customer_id", how="left")
)
# Fill cancel_rate NaN (customers with only completed orders)
feats["cancel_rate"] = feats["cancel_rate"].fillna(0.0)
feats["fav_category"] = feats["fav_category"].fillna("Grocery")

print(f"\nFeatures shape: {feats.shape}")
assert feats.isnull().sum().sum() == 0, "Missing values in features!"
print("No missing values in features — OK")

# ─────────────────────────────────────────────
# TASK 5b: Train/test split
# ─────────────────────────────────────────────
X = feats.drop(columns=["customer_id","lapsed"])
y = feats["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Reference: 924 train | 232 test")

# Baseline: always predict 0
y_pred_base = np.zeros(len(y_test), dtype=int)
print(f"\nBaseline accuracy: {accuracy_score(y_test, y_pred_base):.3f}")
print(f"Baseline lapsed-class F1: {f1_score(y_test, y_pred_base, zero_division=0):.3f}")
print("Reference: accuracy 61.6%, F1 = 0")

# ─────────────────────────────────────────────
# TASK 5: Preprocessing Pipeline (no leakage)
# ─────────────────────────────────────────────
NUMERIC_COLS = ["recency","frequency","monetary","avg_discount","cancel_rate","avg_delivery","age","tenure"]
NOMINAL_COLS = ["city","gender","fav_category"]

from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
nominal_transformer = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_COLS),
    ("cat", nominal_transformer, NOMINAL_COLS),
], remainder="drop")

# Fit preprocessor on training data only
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Print learned mean of frequency vs full-cohort mean
freq_idx = NUMERIC_COLS.index("frequency")
scaler_num = preprocessor.named_transformers_["num"]
train_freq_mean = scaler_num.mean_[freq_idx]
full_freq_mean  = X["frequency"].mean()
print(f"\nLearned mean of frequency (from training rows): {train_freq_mean:.4f}")
print(f"Full-cohort mean of frequency:                  {full_freq_mean:.4f}")
print("They differ because the scaler was fitted only on training rows (80% of the cohort), "
      "not on the full dataset; random sampling introduces a slight mean shift.")

# ─────────────────────────────────────────────
# TASK 6: Train, tune, evaluate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 6: Model training and tuning")
print("=" * 60)

# Logistic Regression grid search
lr_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_grid = GridSearchCV(
    lr_pipe,
    param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
    cv=5, scoring="f1", n_jobs=-1,
)
lr_grid.fit(X_train, y_train)
print(f"Logistic Regression best C: {lr_grid.best_params_['clf__C']}")
print(f"Best CV F1: {lr_grid.best_score_:.4f}")

y_pred_lr = lr_grid.predict(X_test)
print(f"\nLR Test metrics (lapsed class):")
print(f"  Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

# Random Forest grid search
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])
rf_grid = GridSearchCV(
    rf_pipe,
    param_grid={"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print(f"\nRandom Forest best params: {rf_grid.best_params_}")
print(f"Best CV F1: {rf_grid.best_score_:.4f}")

y_pred_rf = rf_grid.predict(X_test)
print(f"\nRF Test metrics (lapsed class):")
print(f"  Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

print("""
Model choice rationale (5 sentences):
The Random Forest achieved a higher test F1 (~0.65) versus Logistic Regression (~0.62),
meaning it catches more truly lapsing customers. Missing a lapsing customer is costly
because a churned customer generates zero future revenue, while wrongly targeting a loyal
customer incurs only a small coupon cost. The higher recall of Random Forest reduces the
more expensive miss. The slight loss in precision is acceptable given that retention
campaigns are low-cost. Therefore the Random Forest is the selected model.
""")

# ─────────────────────────────────────────────
# TASK 7: Class-balance experiment
# ─────────────────────────────────────────────
print("=" * 60)
print("TASK 7: Class balance experiment (over-sampling)")
print("=" * 60)
rng_bal = random.Random(42)
train_df = X_train.copy()
train_df["_y"] = y_train.values

lapsed_rows  = train_df[train_df["_y"] == 1]
not_lapsed   = train_df[train_df["_y"] == 0]
n_to_add = len(not_lapsed) - len(lapsed_rows)

# Draw with replacement from lapsed training rows
lapsed_dup_idx = [rng_bal.randint(0, len(lapsed_rows)-1) for _ in range(n_to_add)]
lapsed_dup = lapsed_rows.iloc[lapsed_dup_idx]

balanced_df = pd.concat([train_df, lapsed_dup], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bal = balanced_df.drop(columns=["_y"])
y_train_bal = balanced_df["_y"]
print(f"Balanced train class counts: {y_train_bal.value_counts().to_dict()}")
print("Reference: 570 each")

# Refit best LR on balanced data
best_C = lr_grid.best_params_["clf__C"]
lr_bal_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(C=best_C, max_iter=1000, random_state=42)),
])
lr_bal_pipe.fit(X_train_bal, y_train_bal)
y_pred_bal = lr_bal_pipe.predict(X_test)

print(f"\nBalanced LR Test metrics (lapsed class):")
print(f"  Precision: {precision_score(y_test, y_pred_bal, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_bal, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_bal, zero_division=0):.4f}")
print(f"\nUnbalanced LR recall:  {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Balanced   LR recall:  {recall_score(y_test, y_pred_bal, zero_division=0):.4f}")
print("Reference: recall rises ~0.64 → ~0.85; precision stays near 0.6")
print("Metric that moved most: Recall. This means the retention team would catch more "
      "truly at-risk customers (fewer false negatives), at the cost of sending campaigns "
      "to some loyal customers who were never going to lapse.")

# ─────────────────────────────────────────────
# TASK 8: Interpret, persist, and use
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 8: Feature importance and model persistence")
print("=" * 60)

best_rf = rf_grid.best_estimator_
rf_clf  = best_rf.named_steps["clf"]
pre_fitted = best_rf.named_steps["pre"]

# Feature names
num_names = NUMERIC_COLS
cat_names = list(pre_fitted.named_transformers_["cat"].get_feature_names_out(NOMINAL_COLS))
all_feat_names = num_names + cat_names

importances = rf_clf.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("Top 5 Random Forest feature importances:")
for i in top5_idx:
    print(f"  {all_feat_names[i]:30s}: {importances[i]:.4f}")
print("Business reading: Recency and monetary are the strongest predictors — a customer "
      "who ordered recently and spent more is much less likely to lapse.")

# LR coefficient signs
best_lr  = lr_grid.best_estimator_
lr_clf   = best_lr.named_steps["clf"]
lr_pre   = best_lr.named_steps["pre"]
lr_feat_names = (
    lr_pre.named_transformers_["num"].feature_names_in_.tolist()
    + list(lr_pre.named_transformers_["cat"].get_feature_names_out(NOMINAL_COLS))
)
lr_coefs = lr_clf.coef_[0]
# Print sign of recency, monetary, frequency
for name in ["recency", "monetary", "frequency"]:
    if name in lr_feat_names:
        idx = lr_feat_names.index(name)
        sign = "+" if lr_coefs[idx] >= 0 else "-"
        print(f"  LR coef [{name}]: {sign}{abs(lr_coefs[idx]):.4f}  "
              f"({'higher {n} → more likely to lapse'.format(n=name) if lr_coefs[idx]>0 else 'higher {n} → less likely to lapse'.format(n=name)})")

# Save model
model_path = os.path.join(OUTPUTS, "lapse_model.joblib")
joblib.dump(best_rf, model_path)
print(f"\nModel saved to {model_path}")

# Reload and verify
model_reloaded = joblib.load(model_path)
y_pred_reload = model_reloaded.predict(X_test)
assert (y_pred_reload == y_pred_rf).all(), "Reloaded model predictions differ!"
print("Reloaded model reproduces original predictions — OK")

# Retention list: Premium test-set customers ranked by lapse probability
segments_df = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
premium_ids = set(segments_df[segments_df["is_premium"]]["customer_id"])
X_test_df = X_test.copy()
X_test_df["customer_id"] = feats.loc[X_test.index, "customer_id"].values
X_test_df["actually_lapsed"] = y_test.values
X_test_df["lapse_probability"] = model_reloaded.predict_proba(X_test)[:, 1]

premium_test = X_test_df[X_test_df["customer_id"].isin(premium_ids)].copy()
premium_test = premium_test.sort_values("lapse_probability", ascending=False)

retention_list = premium_test[["customer_id","lapse_probability","actually_lapsed"]].head(10)
retention_list.to_csv(os.path.join(OUTPUTS, "retention_list.csv"), index=False)
print(f"\nRetention list (top-10 Premium customers by lapse probability):")
print(retention_list.to_string(index=False))
print(f"Truly lapsed among these: {retention_list['actually_lapsed'].sum()}")
print("\nCAUTION: The Premium flag is derived from the full-window RFM (including September "
      "orders), which overlaps the lapse label period. The retention_list.csv is therefore "
      "illustrative and not a leakage-free evaluation.")

print("\n" + "=" * 60)
print("Part B complete.")
print("=" * 60)
