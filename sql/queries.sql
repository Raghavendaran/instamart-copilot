-- sql/queries.sql
-- Seven analytical queries against data/instamart.db
-- All monetary values in INR. Only completed orders count as revenue unless stated otherwise.

-- Q1: Customers whose total high-value completed order spend exceeds INR 8000
-- WHERE: only completed orders with order_amount > 500
-- HAVING: total of those orders > 8000
SELECT
    o.customer_id,
    SUM(o.order_amount) AS high_value_spend
FROM orders o
WHERE o.status = 'completed'
  AND o.order_amount > 500
GROUP BY o.customer_id
HAVING SUM(o.order_amount) > 8000
ORDER BY high_value_spend DESC;

-- Q2: Revenue by city and product category (three-table join)
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Products priced above the average catalogue price (scalar subquery)
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: Customers with no completed order (LEFT JOIN)
SELECT
    c.customer_id,
    c.city,
    c.gender,
    c.age
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customers with total completed spend above the average (CTE)
WITH customer_spend AS (
    SELECT
        customer_id,
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT ROUND(AVG(total_spend), 2) AS avg_total
    FROM customer_spend
)
SELECT
    cs.customer_id,
    cs.total_spend,
    av.avg_total
FROM customer_spend cs, avg_spend av
WHERE cs.total_spend > av.avg_total
ORDER BY cs.total_spend DESC;

-- Q6: Ranking window functions — rank customers by total completed spend within each city
WITH customer_spend AS (
    SELECT
        o.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_spend
ORDER BY city, total_spend DESC;

-- Q7: Daily completed revenue with running total (window function)
SELECT
    SUBSTR(order_ts, 1, 10)                  AS day,
    ROUND(SUM(order_amount), 2)              AS daily_revenue,
    ROUND(SUM(SUM(order_amount)) OVER (
        ORDER BY SUBSTR(order_ts, 1, 10)
    ), 2)                                     AS cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY SUBSTR(order_ts, 1, 10)
ORDER BY day;
