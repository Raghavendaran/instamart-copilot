"""
part_a_validate.py
------------------
Validation script for Part A.
Asserts at least 8 reference checkpoints and prints ALL CHECKPOINTS PASSED if all hold.
"""

import os
import sqlite3
import sys

import pandas as pd

DB = os.path.join("data", "instamart.db")
CLEAN = os.path.join("data", "clean")

errors = []

def check(name, actual, expected, tol=0):
    """Assert actual == expected (within tol for floats). Accumulate errors."""
    if tol:
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    if not ok:
        errors.append(f"  FAIL [{name}]: got {actual}, expected {expected}")
    else:
        print(f"  PASS [{name}]: {actual}")

# ── Load clean data ──────────────────────────────────────────────────────────
customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
orders    = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
products  = pd.read_csv(os.path.join(CLEAN, "products.csv"))

print("Validating Part A checkpoints …\n")

# 1. Orders row count
check("Orders rows after cleaning", len(orders), 12309)

# 2. Completed orders count
completed = orders[orders["status"] == "completed"]
check("Completed orders", len(completed), 11110)

# 3. Cancelled orders count
cancelled = orders[orders["status"] == "cancelled"]
check("Cancelled orders", len(cancelled), 1199)

# 4. No pending rows
check("No pending rows", (orders["status"] == "pending").sum(), 0)

# 5. No missing order_amount
check("No missing order_amount", orders["order_amount"].isna().sum(), 0)

# 6. No missing delivery_minutes among completed
check("No missing delivery_minutes (completed)",
      orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum(), 0)

# 7. Missing delivery_minutes among cancelled == 1199
check("Missing delivery_minutes (cancelled)",
      orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum(), 1199)

# 8. Sum of completed order_amount
completed_sum = round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2)
check("Sum completed order_amount (INR)", completed_sum, 2998668.80, tol=1.0)

# 9. Customer city counts (spot check Pune)
city_counts = customers["city"].value_counts()
check("Pune customer count", int(city_counts.get("Pune", 0)), 227)
check("Delhi customer count", int(city_counts.get("Delhi", 0)), 210)

# 10. Gender counts (spot check female)
gender_counts = customers["gender"].value_counts()
check("Female customer count", int(gender_counts.get("female", 0)), 610)
check("Male customer count",   int(gender_counts.get("male", 0)), 560)

# 11. No missing ages
check("No missing ages", customers["age"].isna().sum(), 0)

# 12. Ages in range 18–60
check("All ages between 18 and 60",
      int(((customers["age"] < 18) | (customers["age"] > 60)).sum()), 0)

# ── SQLite checks ─────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)

# Q1 reference: 30 customers
q1 = pd.read_sql_query("""
    SELECT customer_id, SUM(order_amount) AS high_value_spend
    FROM orders WHERE status='completed' AND order_amount > 500
    GROUP BY customer_id HAVING SUM(order_amount) > 8000
""", conn)
check("Q1 customers above 8000 high-value spend", len(q1), 30)

# Q2 reference: 30 rows, grand total == 2,998,668.80
q2 = pd.read_sql_query("""
    SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue_inr
    FROM orders o
    JOIN customers c ON o.customer_id=c.customer_id
    JOIN products  p ON o.product_id=p.product_id
    WHERE o.status='completed'
    GROUP BY c.city, p.category
""", conn)
check("Q2 rows (city×category)", len(q2), 30)
check("Q2 grand total (INR)", round(q2["revenue_inr"].sum(), 2), 2998668.80, tol=1.0)

# Q3 reference: 11 products
q3 = pd.read_sql_query("""
    SELECT product_id FROM products
    WHERE price_inr > (SELECT AVG(price_inr) FROM products)
""", conn)
check("Q3 products above avg price", len(q3), 11)

# Q4 reference: 25 customers
q4 = pd.read_sql_query("""
    SELECT c.customer_id FROM customers c
    LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
    WHERE o.order_id IS NULL
""", conn)
check("Q4 customers with no completed order", len(q4), 25)

# Q5 reference: 550 customers
q5 = pd.read_sql_query("""
    WITH cs AS (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id),
         av AS (SELECT AVG(total_spend) AS avg_total FROM cs)
    SELECT cs.customer_id FROM cs, av WHERE cs.total_spend > av.avg_total
""", conn)
check("Q5 customers above average spend", len(q5), 550)

# Q7 reference: 183 daily rows
q7 = pd.read_sql_query("""
    SELECT SUBSTR(order_ts,1,10) AS day, SUM(order_amount) AS daily_revenue,
           SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts,1,10)) AS cumulative_revenue
    FROM orders WHERE status='completed'
    GROUP BY SUBSTR(order_ts,1,10) ORDER BY day
""", conn)
check("Q7 daily rows", len(q7), 183)
check("Q7 last cumulative revenue (INR)", round(q7["cumulative_revenue"].iloc[-1], 2), 2998668.80, tol=1.0)

conn.close()

# ── Final result ──────────────────────────────────────────────────────────────
print()
if errors:
    print("CHECKPOINT FAILURES:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("ALL CHECKPOINTS PASSED")
