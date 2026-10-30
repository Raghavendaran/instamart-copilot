"""
part_a_clean.py
---------------
Part A: Data Foundation — Clean, Store, Query
Reads raw CSVs from data/raw/, cleans them, writes to data/clean/ and data/instamart.db.
"""

import os
import re
import sqlite3

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")

os.makedirs(CLEAN, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────
customers_raw = pd.read_csv(os.path.join(RAW, "customers.csv"))
products_raw  = pd.read_csv(os.path.join(RAW, "products.csv"))
orders_raw    = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype=str)
tickets_raw   = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)

print("=" * 60)
print("AUDIT REPORT")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. AUDIT
# ─────────────────────────────────────────────
print(f"\norders.csv raw rows              : {len(orders_raw)}")

exact_dups = orders_raw.duplicated().sum()
print(f"Exact duplicate rows             : {exact_dups}")

pending_rows = (orders_raw["status"] == "pending").sum()
print(f"Rows with status=pending         : {pending_rows}")

distinct_order_ids = orders_raw["order_id"].nunique()
print(f"Distinct order_id values         : {distinct_order_ids}")

non_numeric_amount = pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum()
print(f"Non-numeric order_amount entries : {non_numeric_amount}")

neg_delivery = (pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce") < 0).sum()
print(f"Negative delivery_minutes        : {neg_delivery}")

blank_delivery = (orders_raw["delivery_minutes"].fillna("").str.strip() == "").sum()
print(f"Blank delivery_minutes           : {blank_delivery}")

# Customers audit
cust_raw = customers_raw.copy()
blank_age    = (cust_raw["age"].fillna("").astype(str).str.strip() == "").sum()
print(f"\nBlank age                        : {blank_age}")

age_numeric = pd.to_numeric(cust_raw["age"], errors="coerce")
out_of_range = ((age_numeric < 10) | (age_numeric > 100)).sum()
print(f"Age values outside 10-100        : {out_of_range}")

raw_city_spellings = cust_raw["city"].nunique()
print(f"Distinct raw city spellings      : {raw_city_spellings}")

raw_gender_nonblank = cust_raw[cust_raw["gender"].fillna("").astype(str).str.strip() != ""]["gender"].nunique()
print(f"Distinct non-blank raw gender spellings: {raw_gender_nonblank}")

blank_gender = (cust_raw["gender"].fillna("").astype(str).str.strip() == "").sum()
print(f"Blank gender                     : {blank_gender}")

print("\n" + "=" * 60)
print("CLEANING")
print("=" * 60)

# ─────────────────────────────────────────────
# 3. CLEAN CUSTOMERS
# ─────────────────────────────────────────────
customers = customers_raw.copy()

# (a) Standardise city: strip whitespace + title case → exactly 6 cities
customers["city"] = customers["city"].str.strip().str.title()
print(f"\nCity value counts after cleaning:\n{customers['city'].value_counts().to_string()}")

# (b) Map gender spellings → male / female / unknown
GENDER_MAP = {
    "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
    "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female",
}
customers["gender"] = customers["gender"].astype(str).str.strip().map(GENDER_MAP).fillna("unknown")
print(f"\nGender value counts after cleaning:\n{customers['gender'].value_counts().to_string()}")

# (c) Age: mark impossible ages (outside 10–100) as NaN, then fill with city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age).round().astype(int)
print(f"\nMissing ages after cleaning      : {customers['age'].isna().sum()}")
print(f"Age range: {customers['age'].min()} – {customers['age'].max()}")

# (d) Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print(f"\nCustomers shape after cleaning   : {customers.shape}")

# ─────────────────────────────────────────────
# 4. CLEAN ORDERS
# ─────────────────────────────────────────────
orders = orders_raw.copy()

# Step 1: Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
print(f"\nOrders before dedup              : {len(orders)}")

# Step 2: Remove exact duplicate rows
orders = orders.drop_duplicates()
print(f"Orders after removing exact dups : {len(orders)}")

# Step 3: For each order_id, keep only the record with the latest order_ts
#         (handles pending + later final-status pairs)
orders = orders.sort_values("order_ts", ascending=False)
orders = orders.drop_duplicates(subset=["order_id"], keep="first")
orders = orders.sort_values("order_id").reset_index(drop=True)
print(f"Orders after keeping latest per order_id: {len(orders)}")

# Step 4: Convert order_amount to numeric; fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
products = products_raw.copy()
price_map = products.set_index("product_id")["price_inr"].to_dict()

orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")

missing_amount_mask = orders["order_amount"].isna()
print(f"Missing order_amount before fill : {missing_amount_mask.sum()}")
orders.loc[missing_amount_mask, "order_amount"] = orders.loc[missing_amount_mask].apply(
    lambda row: round(price_map.get(row["product_id"], 0) * row["quantity"] * (1 - row["discount_pct"] / 100), 2),
    axis=1
)
print(f"Missing order_amount after fill  : {orders['order_amount'].isna().sum()}")

# Step 5: Handle delivery_minutes
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

# Set negative values to NaN
neg_mask = orders["delivery_minutes"] < 0
print(f"Negative delivery_minutes set to NaN: {neg_mask.sum()}")
orders.loc[neg_mask, "delivery_minutes"] = np.nan

# Join with products to get category
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# Fill missing delivery_minutes for completed orders with category median
completed_mask = orders["status"] == "completed"
cat_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)
orders.loc[completed_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = \
    cat_median[completed_mask & orders["delivery_minutes"].isna()]

# Cancelled orders: leave delivery_minutes as NaN
print(f"Missing delivery_minutes (completed) after fill: {orders.loc[completed_mask, 'delivery_minutes'].isna().sum()}")
print(f"Missing delivery_minutes (cancelled): {orders.loc[~completed_mask, 'delivery_minutes'].isna().sum()}")

print(f"\nFinal orders shape               : {orders.shape}")
print(f"Status counts:\n{orders['status'].value_counts().to_string()}")

completed_sum = orders.loc[completed_mask, "order_amount"].sum()
print(f"\nSum of order_amount (completed)  : INR {completed_sum:,.2f}")

# ─────────────────────────────────────────────
# 5. WRITE CLEAN CSVs
# ─────────────────────────────────────────────
# Convert signup_date and order_ts back to strings for CSV
customers_save = customers.copy()
customers_save["signup_date"] = customers_save["signup_date"].dt.strftime("%Y-%m-%d")
customers_save.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)

orders_save = orders.copy()
orders_save["order_ts"] = orders_save["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders_save.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)

print(f"\nClean CSVs written to {CLEAN}/")

# ─────────────────────────────────────────────
# 6. STORE IN SQLITE
# ─────────────────────────────────────────────
if os.path.exists(DB):
    os.remove(DB)

conn = sqlite3.connect(DB)

# Customers: store signup_date as text YYYY-MM-DD
cust_db = customers.copy()
cust_db["signup_date"] = cust_db["signup_date"].dt.strftime("%Y-%m-%d")
cust_db.to_sql("customers", conn, if_exists="replace", index=False)

# Products
products.to_sql("products", conn, if_exists="replace", index=False)

# Orders: store order_ts as text YYYY-MM-DD HH:MM:SS
# Keep category column dropped (it's derivable via join)
orders_db = orders.copy()
orders_db["order_ts"] = orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders_db = orders_db.drop(columns=["category"])
orders_db.to_sql("orders", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
print(f"SQLite database written to {DB}")

print("\n" + "=" * 60)
print("Part A cleaning complete.")
print("=" * 60)
