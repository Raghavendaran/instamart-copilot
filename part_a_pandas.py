"""
part_a_pandas.py
----------------
Part A Task 7: pandas cross-checks and exploratory analysis.
"""

import os
import numpy as np
import pandas as pd

CLEAN = os.path.join("data", "clean")

orders    = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
products  = pd.read_csv(os.path.join(CLEAN, "products.csv"))

# Merge for pivot — category is already in orders.csv from Part A cleaning
orders_full = orders.merge(customers[["customer_id","city"]], on="customer_id", how="left")
if "category" not in orders_full.columns:
    orders_full = orders_full.merge(products[["product_id","category"]], on="product_id", how="left")
completed = orders_full[orders_full["status"] == "completed"].copy()

print("=" * 60)
print("7a: Pivot table — Revenue by city and category")
print("=" * 60)
pivot = pd.pivot_table(
    completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)
print(pivot.round(2).to_string())
grand_total = pivot.loc["All", "All"]
print(f"\nGrand total from pivot: INR {grand_total:,.2f}")
print(f"Reference:              INR 2,998,668.80\n")

print("=" * 60)
print("7a: groupby — sum, mean, count of completed order_amount per city")
print("=" * 60)
city_agg = (
    completed
    .groupby("city")["order_amount"]
    .agg(sum="sum", mean="mean", count="count")
    .round(2)
)
print(city_agg.to_string())
print(f"\nReference Bengaluru: sum=523,643.40 | mean=276.47 | count=1,894")
print(f"Reference Mumbai:    sum=479,058.50 | mean=297.00 | count=1,613\n")

print("=" * 60)
print("7b: Descriptive statistics of completed order_amount")
print("=" * 60)
oa = completed["order_amount"]
mean_oa   = oa.mean()
median_oa = oa.median()
skew_oa   = oa.skew()
q1        = oa.quantile(0.25)
q3        = oa.quantile(0.75)
iqr       = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
out_above = (oa > upper_fence).sum()
out_below = (oa < lower_fence).sum()

print(f"Mean                 : INR {mean_oa:.2f}")
print(f"Median               : INR {median_oa:.2f}")
print(f"Skewness             : {skew_oa:.2f}")
print(f"Q1                   : INR {q1:.2f}")
print(f"Q3                   : INR {q3:.2f}")
print(f"IQR                  : INR {iqr:.2f}")
print(f"Lower fence (Q1-1.5×IQR): INR {lower_fence:.2f}")
print(f"Upper fence (Q3+1.5×IQR): INR {upper_fence:.2f}")
print(f"Outliers above upper fence: {out_above}")
print(f"Outliers below lower fence: {out_below}")

print(f"""
Reference:
  Mean 269.91 | Median 135.00 | Skewness ~4.61
  Q1 64.80    | Q3 297.00     | Upper fence 645.30
  Outliers above 897 | Below 0
""")

print("=" * 60)
print("Correlation matrix (completed orders)")
print("=" * 60)
corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr_data = completed[corr_cols].dropna()
corr_matrix = corr_data.corr()
print(corr_matrix.round(3).to_string())

# Find pair with largest absolute correlation (excluding diagonal)
corr_abs = corr_matrix.abs()
np.fill_diagonal(corr_abs.values, 0)
# upper triangle only
for i in range(len(corr_abs)):
    for j in range(i):
        corr_abs.iloc[i, j] = 0
max_pair = corr_abs.stack().idxmax()
max_val  = corr_matrix.loc[max_pair[0], max_pair[1]]
print(f"\nLargest absolute correlation: {max_pair[0]} vs {max_pair[1]} = {max_val:.3f}")
print("Reference: order_amount vs discount_pct ≈ -0.095")
print("Note: correlation of this size does not establish causation.\n")
