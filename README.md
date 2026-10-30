# Instamart Customer Intelligence & Support Copilot

An end-to-end data and AI pipeline for Swiggy Instamart's Customer Experience team.  
Built across four interconnected Parts: **Data Foundation → Customer Segments → Ticket Triage → Policy RAG & Refund Agent**.

All data is **synthetic and fictional**, generated deterministically by `generate_data.py`.  
No paid API, no network calls, no API keys required — every LLM call goes through the offline `mock_llm.py`.  
All monetary amounts are in **INR (Indian Rupees)**.

---

## Overview

| Part | Title | Marks |
|------|-------|-------|
| A | Data Foundation: Clean, Store, Query | 22 |
| B | Customer Segments and Lapse Prediction | 28 |
| C | Support Ticket Triage with Guardrails | 22 |
| D | Policy RAG Assistant and Refund Agent | 28 |

---

## Reproduce

Run these commands in order from the repository root in a clean clone.  
Python 3.11+ and SQLite 3.25+ are required.

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate raw data
python generate_data.py
# Expected last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164

# 5. Part A — Clean data, build SQLite DB
python part_a_clean.py

# 6. Part A — Run SQL queries
python part_a_sql.py

# 7. Part A — Pandas cross-check & exploratory analysis
python part_a_pandas.py

# 8. Part A — Validate all checkpoints
python part_a_validate.py
# Expected output: ALL CHECKPOINTS PASSED

# 9. Part B — Customer segments and lapse model
python part_b_segments.py

# 10. Part C — Support ticket triage
python part_c_triage.py

# 11. Part D — Policy RAG and refund agent
python part_d_rag.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### Audit Results

| Metric | Value | Reference |
|--------|-------|-----------|
| Raw orders.csv rows | 13,084 | 13,084 ✓ |
| Exact duplicate rows | 119 | 119 ✓ |
| Rows with status=pending | 656 | 656 ✓ |
| Distinct order_id values | 12,309 | 12,309 ✓ |
| Non-numeric order_amount entries | 120 | 120 ✓ |
| Negative delivery_minutes | 106 | 106 ✓ |
| Blank delivery_minutes | 2,541 | 2,541 ✓ |
| Blank age values | 61 | 61 ✓ |
| Age values outside 10–100 | 6 | 6 ✓ |
| Distinct raw city spellings | 24 | 24 ✓ |
| Distinct non-blank raw gender spellings | 10 | 10 ✓ |
| Blank gender values | 30 | 30 ✓ |

### Cleaning Rules

**Customers:**

1. **City standardisation** — Strip leading/trailing whitespace, apply `.str.title()`. Changed 24 raw spellings → 6 canonical cities.
2. **Gender mapping** — Map `{male, Male, M, m, MALE} → male`, `{female, Female, F, f, FEMALE} → female`, blank/unknown → `unknown`. Changed 10 raw spellings → 3 values.
3. **Age imputation** — Values outside 10–100 (6 rows) set to NaN. Blank ages (61 rows) also set to NaN. All 67 NaN ages filled with the median age of the customer's city (groupby + transform).
4. **signup_date** — Converted from string to `datetime64`.

| City | Count |
|------|-------|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Count |
|--------|-------|
| female | 610 |
| male | 560 |
| unknown | 30 |

**Orders (applied in order):**

1. **order_ts to datetime** — Converted `order_ts` string column to `datetime64`.
2. **Remove exact duplicates** — Dropped 119 exact duplicate rows (13,084 → 12,965).
3. **Keep latest record per order_id** — For each `order_id` with multiple records (pending + final status), kept the row with the latest `order_ts`. Reduced to 12,309 rows (one row per `order_id`). Eliminated all 656 pending rows.
4. **order_amount imputation** — Converted to numeric; 120 non-parseable values became NaN, then filled as `round(price_inr × quantity × (1 − discount_pct / 100), 2)` using the product price from `products.csv`. 0 NaN remaining.
5. **delivery_minutes cleaning** — 106 negative values set to NaN. For completed orders, remaining NaN delivery times filled with the median delivery time of the order's product category (groupby + transform). Cancelled orders left as NaN — a cancelled order was never delivered, so imputing a delivery time would invent data.

### Post-cleaning Counts

| Metric | Value | Reference |
|--------|-------|-----------|
| Total orders | 12,309 | 12,309 ✓ |
| Completed orders | 11,110 | 11,110 ✓ |
| Cancelled orders | 1,199 | 1,199 ✓ |
| Pending orders | 0 | 0 ✓ |
| Missing order_amount | 0 | 0 ✓ |
| Missing delivery_minutes (completed) | 0 | 0 ✓ |
| Missing delivery_minutes (cancelled) | 1,199 | 1,199 ✓ |
| Sum of completed order_amount | INR 2,998,668.80 | INR 2,998,668.80 ✓ |

### SQL Query Results

**Q1 — WHERE then HAVING**  
Customers with completed orders >INR 500 and total of those orders >INR 8,000: **30 customers**.

**Q2 — Three-table join: Revenue by city and product category**  
30 rows; grand total INR 2,998,668.80; largest cell: Bengaluru / Electronics = INR 233,909.45.

**Q3 — Scalar subquery: Products above average price**  
Average catalogue price: INR 399.58. Products above average: **11**.

**Q4 — LEFT JOIN: Customers with no completed order**  
**25 customers** have no completed order.

**Q5 — CTE: Customers above average spend**  
Average completed spend per customer: **INR 2,552.06**. Customers above average: **550**.

**Q6 — Ranking window functions**

| City | Customer | Total Spend (INR) |
|------|----------|-------------------|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

**8 rows** have `RANK()` ≠ `ROW_NUMBER()`. This happens when two or more customers have the same total spend within a city: `RANK()` assigns them the same rank and leaves a gap (both get rank 3, next gets rank 5), while `ROW_NUMBER()` arbitrarily breaks the tie and increments sequentially, so they receive different row numbers.

**Q7 — Running total**  
183 daily rows; last cumulative revenue = **INR 2,998,668.80**.

### Pandas Cross-check and Exploratory Analysis

**Q2 pivot grand total:** INR 2,998,668.80 (matches SQL) ✓

**City-level completed order stats:**

| City | Sum (INR) | Mean (INR) | Count |
|------|-----------|------------|-------|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | — | — | — |
| Delhi | — | — | — |
| Hyderabad | — | — | — |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | — | — | — |

**Descriptive statistics of completed `order_amount`:**

| Statistic | Value (INR) |
|-----------|-------------|
| Mean | 269.91 |
| Median | 135.00 |
| Skewness | ~4.61 |
| Q1 | 64.80 |
| Q3 | 297.00 |
| Lower fence (Q1 − 1.5×IQR) | −308.10 |
| Upper fence (Q3 + 1.5×IQR) | 645.30 |
| Outliers above upper fence | 897 |
| Outliers below lower fence | 0 |

**What is this data about?**  
Completed order amounts from Instamart's six metro cities over a 6-month window (April–September 2025), covering groceries, dairy, snacks, personal care and electronics.

**What do its distributions and statistics look like?**  
The distribution is strongly right-skewed (skewness ≈ 4.61): median (INR 135) is much lower than mean (INR 269.91), driven by a long upper tail of high-value electronics orders. Most orders cluster in the INR 20–300 band.

**Are there outliers or anomalies?**  
897 orders lie above the 1.5×IQR upper fence of INR 645.30, all legitimate electronics purchases (e.g., earbuds at INR 1,499). No outliers below the lower fence.

**What relationships exist between variables?**  
Correlation matrix (completed orders): `order_amount` vs `discount_pct` ≈ −0.095 (largest absolute value); `order_amount` vs `quantity` ≈ +0.087. All correlations are small, indicating weak linear relationships. A correlation of this size does not establish causation — higher discounts may attract higher-priced items or vice versa, but the causal direction cannot be inferred from correlation alone.

---

## Part B — Customer Segments and Lapse Prediction

### RFM Table

Snapshot: 2025-10-01 00:00:00. Using only completed orders.

| Metric | Value | Reference |
|--------|-------|-----------|
| Customers in RFM | 1,175 | 1,175 ✓ |
| Recency median | 19 days | 19 ✓ |
| Recency min | 0 days | 0 ✓ |
| Recency max | 180 days | 180 ✓ |
| Frequency mean | 9.46 | 9.46 ✓ |
| Frequency max | 33 | 33 ✓ |
| Monetary mean | INR 2,552.06 | INR 2,552.06 ✓ |
| Monetary max | INR 14,766.55 | INR 14,766.55 ✓ |

### K-means Clustering (K = 2 to 8)

RFM columns standardised with `StandardScaler` before clustering.

| K | Inertia | Silhouette |
|---|---------|------------|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | **0.526** |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Why K=8 always has lower inertia than K=2 but cannot be chosen on that basis alone:**  
Inertia (within-cluster sum of squares) monotonically decreases as K increases because adding more clusters always reduces average within-cluster distance — at the extreme, K=N gives inertia = 0. Inertia alone therefore cannot pick K; it must be balanced against interpretability and cluster separation (silhouette). The silhouette score peaks at K=4 (0.526), and the inertia curve shows a clear elbow between K=3 and K=5. K=4 also produces four business-interpretable customer groups, making it the practical choice.

**Chosen K: 4**

### Segment Profiles

| Segment | Name | Size | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Recommended Action |
|---------|------|------|---------------------|----------------|---------------------|--------------------|
| Highest monetary | **Premium** | ~138 | high | ~6.8 | ~7,130 | Exclusive early-access sales and premium loyalty perks |
| Highest frequency | Active Loyalist | ~419 | ~8.8 | ~19.4 | medium | Personalised reorder reminders and bundle offers |
| Medium everything | Occasional Buyer | ~436 | medium | medium | medium | Targeted discount coupons for favourite category |
| High recency | At-Risk | ~182 | high | low | low | Win-back campaign with time-limited incentive |

> **Note:** Cluster IDs (0, 1, 2, 3) are arbitrary labels assigned by K-means and carry no inherent ordering or ranking.

`is_premium = True` for the segment with the highest mean monetary value (≈ INR 7,130).

### PCA Projection

`PCA(n_components=2)` applied to the scaled RFM matrix:

| Component | Explained Variance Ratio |
|-----------|--------------------------|
| PC1 | ~0.634 |
| PC2 | ~0.218 |
| **Total** | **~0.852** |

### Lapse Problem Definition

Cutoff: 2025-09-01. Cohort: customers with ≥1 completed order before the cutoff.  
Label: `lapsed=1` if no completed order in 2025-09-01 through 2025-09-30.

| Metric | Value | Reference |
|--------|-------|-----------|
| Cohort size | 1,156 | 1,156 ✓ |
| Lapsed | 443 (38.32%) | 443 (38.32%) ✓ |
| Training rows | 924 | 924 ✓ |
| Test rows | 232 | 232 ✓ |

**Baseline (always predict "not lapsed"):** accuracy 61.6%, lapsed-class F1 = 0.

### Feature Engineering

Features built using **only orders dated before 2025-09-01** (no leakage):

| Feature | Description |
|---------|-------------|
| recency | Days from cutoff to last completed order |
| frequency | Count of distinct completed orders |
| monetary | Sum of completed order amounts |
| avg_discount | Mean discount_pct of completed orders |
| cancel_rate | Cancelled ÷ all final-status orders |
| avg_delivery | Mean delivery_minutes of completed orders |
| fav_category | Most frequent product category ordered |
| city | Customer city (one-hot encoded, k−1 columns) |
| age | Customer age |
| gender | Customer gender (one-hot encoded, k−1 columns) |
| tenure | Days from signup_date to cutoff |

**Segment labels from Task 3 were NOT used as features** — that would constitute leakage.  
**No order data from 2025-09-01 or later was used** — that would reveal the lapse label.

### Preprocessing (No Leakage)

`OneHotEncoder(drop="first")` and `StandardScaler` fitted **on training rows only** inside a scikit-learn `Pipeline`, then applied to test rows.

The learned mean of `frequency` from the fitted scaler differs slightly from the full-cohort mean because the scaler was fitted only on the 80% training split; random sampling introduces a small mean shift.

### Model Training and Evaluation

| Model | Best Params | CV F1 | Test Precision | Test Recall | Test F1 |
|-------|-------------|-------|----------------|-------------|---------|
| Logistic Regression | C=1 | ~0.62 | ~0.62 | ~0.64 | ~0.62 |
| Random Forest | depth=8, leaf=10 | ~0.65 | ~0.64 | ~0.67 | ~0.65 |
| **Baseline** | — | — | — | — | 0.00 |

**Selected model: Random Forest** — It achieves a higher test F1 (~0.65) and recall, catching more truly lapsing customers. In the retention context, missing a lapsing customer (false negative) means losing their entire future revenue, while wrongly targeting a loyal customer (false positive) costs only a small coupon. Therefore higher recall is more valuable, and the Random Forest's higher recall justifies the choice. The interpretable feature importances also help the business understand what drives churn.

### Class-Balance Experiment

Over-sampled lapsed training rows (with replacement, `random_state=42`) to balance classes (570 each). Refitted best Logistic Regression (C=1) on balanced data.

| Metric | Unbalanced LR | Balanced LR |
|--------|--------------|-------------|
| Precision | ~0.62 | ~0.60 |
| Recall | ~0.64 | ~0.85 |
| F1 | ~0.62 | ~0.70 |

**Metric that moved most: Recall** (+0.21). This means the retention team would identify significantly more truly at-risk customers, at the cost of sending campaigns to some loyal customers who were not about to lapse. The precision remained near 0.6 (it did not rise appreciably). For a low-cost retention campaign, this trade-off is acceptable.

### Feature Importance

**Top 5 Random Forest features** (`feature_importances_`):

| Feature | Importance | Business Reading |
|---------|-----------|-----------------|
| recency | highest | Customers who ordered recently are far less likely to lapse |
| monetary | 2nd | Higher cumulative spend signals stronger engagement |
| frequency | 3rd | More frequent buyers are more habitual and less likely to churn |
| tenure | 4th | Longer-tenured customers have established habits |
| avg_delivery | 5th | Slow deliveries may push customers away |

**Logistic Regression coefficient signs** (best model, C=1):

| Feature | Sign | Reading |
|---------|------|---------|
| recency | + | Higher recency (ordered less recently) → more likely to lapse |
| monetary | − | Higher spend → less likely to lapse |
| frequency | − | More orders → less likely to lapse |

### Model Persistence

Model saved to `outputs/lapse_model.joblib` with `joblib.dump`.  
Reloaded with `joblib.load` and confirmed to reproduce identical test predictions.

### Retention List

Top-10 Premium-segment test-set customers ranked by predicted lapse probability saved to `outputs/retention_list.csv` (columns: `customer_id`, `lapse_probability`, `actually_lapsed`).

> **IMPORTANT — Leakage caveat:** The `is_premium` flag is derived from the full-window RFM (Tasks 1–3), which includes September 2025 orders that overlap with the lapse label period. The `retention_list.csv` is therefore **illustrative only** and does not constitute a leakage-free evaluation. The model's own features strictly use only pre-cutoff data as required.

---

## Part C — Support Ticket Triage with Guardrails

### Input Guardrails

| Filter | Count | Reference |
|--------|-------|-----------|
| Empty/whitespace tickets (skipped_empty) | 4 | 4 ✓ |
| Long tickets >400 tokens (flagged_long) | 4 | 4 ✓ |
| Long ticket token estimate | ~734 tokens | 734 ✓ |

### PII Redaction

Applied in order: email → phone → order ID.

| Pattern | Substitutions | Reference |
|---------|--------------|-----------|
| Email (`[\w.+-]+@[\w-]+\.[\w.]+`) → `[EMAIL]` | 55 | 55 ✓ |
| Phone (`\b[6-9]\d{9}\b`) → `[PHONE]` | 91 | 91 ✓ |
| Order ID (`ORD-\d{6}`) → `[ORDER_ID]` | 132 | 132 ✓ |
| **Total** | **278** | **278 ✓** |

After redaction, re-applying all three patterns finds **0 matches** (verified in code).

### Prompts

Two system prompts are used:

- **Zero-shot** (`ZERO_SHOT_SYSTEM`): Contains the word `CLASSIFY`, lists 4 categories, requires JSON-only reply. Does **not** contain `EXAMPLES:`.
- **Few-shot** (`FEW_SHOT_SYSTEM`): Contains `CLASSIFY`, lists 4 categories, requires JSON reply, and includes a line `EXAMPLES:` followed by four labelled examples including one fraud example.

### Resilient Calls

`TransientLLMError` is retried up to 3 times per ticket with exponential backoff + jitter:  
wait = `0.05 × 2^n × Uniform(0.9, 1.15)` where n ∈ {0, 1, 2}.

A full run over 156 non-empty, non-long tickets produces exactly **19 TransientLLMErrors**, all recovered by retry. No ticket ends with a permanent error.

### Output Validation

| Status | Meaning |
|--------|---------|
| `classified_json` | Valid JSON with allowed category |
| `extracted_text` | Non-JSON reply; category extracted by regex |
| `fallback_technical` | No category found; defaulted to "technical" |

### Routing

- `fraud` tickets → **high priority**
- Premium-segment customers → **high priority**
- All others → **standard priority**

Results saved to `outputs/triage_results.csv`.

### Token Budget Demo

A budget of 200 tokens per ticket is demonstrated. Tickets where the estimated prompt exceeds the budget are marked `budget_exceeded` without calling the model.  
Results saved to `outputs/triage_budget_demo.csv`.

---

## Part D — Policy RAG Assistant and Refund Agent

### Policy RAG Assistant

Five policy documents indexed from `data/policies/`:
- `refund_policy.txt`, `delivery_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, `escalation_policy.txt`

Retrieval uses TF-IDF-style stem overlap (matching `mock_llm._stems`). The two most relevant documents are included as context with filename tags.  
Questions with `expected_doc = NONE` receive no context and trigger the "not found" response.

**Evaluation on `policy_eval.csv` (14 questions):**

| Metric | Result |
|--------|--------|
| Document retrieval accuracy | ~85% |
| Fact extraction accuracy (answerable questions) | ~83% |

### Refund Agent — Seven-Scenario Results

| Scenario | Customer | Message | Expected | Actual | Refund (INR) | Match |
|----------|----------|---------|----------|--------|--------------|-------|
| S1 | — | Late delivery, small order 1 | REFUND | REFUND | ≤INR 1,000 | ✓ |
| S2 | — | Late delivery, small order 2 | REFUND | REFUND | ≤INR 1,000 | ✓ |
| S3 | — | Late delivery, large order | ESCALATE | ESCALATE | >INR 1,000 | ✓ |
| S4 | — | On-time delivery | DENIED | DENIED | 0 | ✓ |
| S5 | — | Cancelled order | DENIED | DENIED | 0 | ✓ |
| S6 | C0001 | Non-existent order ORD-999999 | NOT_FOUND | NOT_FOUND | 0 | ✓ |
| S7 | — | [LOOP] message | FALLBACK | FALLBACK | 0 | ✓ |

### Stop-Safely Demonstration (S7 FALLBACK)

Scenario S7 contains `[LOOP]`, causing the agent to repeatedly call `search_policy`. The agent detects the loop (>3 `search_policy` calls) and terminates cleanly with outcome `FALLBACK`, preventing an infinite loop.

### Crash-and-Resume Demonstration (S1)

1. Agent runs S1 and executes `get_order` in step 0. State saved to `state/S1_resume.json`.
2. Crash simulated (execution stopped after step 0).
3. State loaded from `state/S1_resume.json`; agent resumes from step 1.
4. Agent completes the full sequence: `get_order → search_policy → calculate_refund → issue_refund`.

**Full tool sequence:** `['get_order', 'search_policy', 'calculate_refund', 'issue_refund']`

Results saved to:
- `outputs/agent_runs.jsonl` — one record per scenario
- `outputs/refund_ledger.jsonl` — one record per approved/escalated refund
- `state/` — persisted conversation state files

---

## Limitations and Next Steps

1. **Synthetic data** — All data is fictional and generated by a deterministic script. Real-world distributions of order values, delivery times and customer behaviour would differ significantly.

2. **Offline LLM mock** — `mock_llm.py` uses keyword matching and sentence scoring, not a real language model. Production deployment would require a hosted model (GPT-4, Gemini, etc.) and would incur API costs.

3. **Retention list leakage caveat** — The `is_premium` flag (used to filter the retention list) is derived from the full-window RFM including September 2025 orders, which overlap with the lapse label period. The `retention_list.csv` is **illustrative only** and not a leakage-free evaluation. In production, segment labels should be computed from pre-cutoff data only.

4. **Lapse model features** — The model uses a snapshot of pre-cutoff behaviour only. Additional signals (app engagement, browsing history, support ticket frequency) could improve prediction.

5. **Triage accuracy** — The mock LLM's keyword-based classification is deterministic and simplified. A real model would handle ambiguous tickets more gracefully (e.g., a billing dispute that also involves fraud).

6. **Policy RAG retrieval** — The current retriever uses stem overlap without embeddings. Semantic retrieval (e.g., sentence-transformers + FAISS) would improve accuracy on paraphrased or synonym-heavy questions.

7. **Agent robustness** — The refund agent currently handles only the specified tool sequence. Real agents need error handling for database connectivity, partial results and ambiguous order matches.

8. **Scalability** — The current SQLite database and in-memory pandas pipeline are adequate for 12,000 orders. A production system would need a distributed database and streaming pipeline for millions of daily orders.
