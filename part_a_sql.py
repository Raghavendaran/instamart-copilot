"""
part_a_sql.py
-------------
Runs all seven SQL queries from sql/queries.sql against data/instamart.db
and prints results for README reporting.
"""

import os
import sqlite3
import pandas as pd

DB = os.path.join("data", "instamart.db")
conn = sqlite3.connect(DB)

print("=" * 60)
print("Q1 â€” WHERE then HAVING: customers with high-value spend > INR 8000")
print("=" * 60)
q1 = pd.read_sql_query("""
    SELECT customer_id, ROUND(SUM(order_amount),2) AS high_value_spend
    FROM orders WHERE status='completed' AND order_amount > 500
    GROUP BY customer_id HAVING SUM(order_amount) > 8000
    ORDER BY high_value_spend DESC
""", conn)
print(q1.to_string(index=False))
print(f"\nReference: 30 customers | Got: {len(q1)} customers\n")

print("=" * 60)
print("Q2 â€” Three-table join: Revenue by city and product category")
print("=" * 60)
q2 = pd.read_sql_query("""
    SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue_inr
    FROM orders o
    JOIN customers c ON o.customer_id=c.customer_id
    JOIN products  p ON o.product_id=p.product_id
    WHERE o.status='completed'
    GROUP BY c.city, p.category
    ORDER BY c.city, p.category
""", conn)
print(q2.to_string(index=False))
print(f"\nRows: {len(q2)} | Grand total: INR {q2['revenue_inr'].sum():,.2f}")
bengaluru_elec = q2[(q2["city"]=="Bengaluru") & (q2["category"]=="Electronics")]["revenue_inr"].values
print(f"Bengaluru/Electronics: INR {bengaluru_elec[0]:,.2f}")
print(f"Reference: 30 rows | INR 2,998,668.80 | Bengaluru/Electronics: INR 233,909.45\n")

print("=" * 60)
print("Q3 â€” Scalar subquery: Products above average price")
print("=" * 60)
q3 = pd.read_sql_query("""
    SELECT product_id, product_name, category, price_inr
    FROM products
    WHERE price_inr > (SELECT AVG(price_inr) FROM products)
    ORDER BY price_inr DESC
""", conn)
print(q3.to_string(index=False))
print(f"\nReference: 11 products | Got: {len(q3)}\n")

print("=" * 60)
print("Q4 â€” LEFT JOIN: Customers with no completed order")
print("=" * 60)
q4 = pd.read_sql_query("""
    SELECT c.customer_id, c.city, c.gender, c.age
    FROM customers c
    LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
    WHERE o.order_id IS NULL
    ORDER BY c.customer_id
""", conn)
print(q4.to_string(index=False))
print(f"\nReference: 25 customers | Got: {len(q4)}\n")

print("=" * 60)
print("Q5 â€” CTE: Customers above average spend")
print("=" * 60)
q5 = pd.read_sql_query("""
    WITH cs AS (
        SELECT customer_id, ROUND(SUM(order_amount),2) AS total_spend
        FROM orders WHERE status='completed' GROUP BY customer_id
    ),
    av AS (SELECT ROUND(AVG(total_spend),2) AS avg_total FROM cs)
    SELECT cs.customer_id, cs.total_spend, av.avg_total
    FROM cs, av WHERE cs.total_spend > av.avg_total
    ORDER BY cs.total_spend DESC
""", conn)
avg_val = q5["avg_total"].iloc[0] if len(q5) > 0 else 0
print(f"Customers above average: {len(q5)} | Average spend: INR {avg_val:,.2f}")
print(f"Reference: 550 customers | Average INR 2,552.06\n")

print("=" * 60)
print("Q6 â€” Window functions: Ranking customers within city by spend")
print("=" * 60)
q6 = pd.read_sql_query("""
    WITH cs AS (
        SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount),2) AS total_spend
        FROM orders o JOIN customers c ON o.customer_id=c.customer_id
        WHERE o.status='completed'
        GROUP BY o.customer_id, c.city
    )
    SELECT customer_id, city, total_spend,
           RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
           DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
           ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM cs ORDER BY city, total_spend DESC
""", conn)
top_per_city = q6.groupby("city").first().reset_index()[["city","customer_id","total_spend"]]
print("Top customer per city:")
print(top_per_city.to_string(index=False))
rows_rank_diff = (q6["rnk"] != q6["row_num"]).sum()
print(f"\nRows where RANK() â‰  ROW_NUMBER(): {rows_rank_diff}")
print(f"Reference: 8 rows\n")

print("=" * 60)
print("Q7 â€” Running total: Daily completed revenue")
print("=" * 60)
q7 = pd.read_sql_query("""
    SELECT SUBSTR(order_ts,1,10) AS day,
           ROUND(SUM(order_amount),2) AS daily_revenue,
           ROUND(SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts,1,10)),2) AS cumulative_revenue
    FROM orders WHERE status='completed'
    GROUP BY SUBSTR(order_ts,1,10) ORDER BY day
""", conn)
print(f"Daily rows: {len(q7)}")
print(f"Last cumulative revenue: INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")
print(f"Reference: 183 rows | INR 2,998,668.80\n")
print(q7.tail(5).to_string(index=False))

conn.close()

print("\n" + "=" * 60)
print("SQL analysis complete.")
print("=" * 60)


