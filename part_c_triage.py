"""
part_c_triage.py
----------------
Part C: Support Ticket Triage with Guardrails.
Reads tickets from data/raw/tickets.csv and customer_segments.csv from Part B.
Outputs triage_results.csv and triage_budget_demo.csv.
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import logging
import math
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError

OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_INPUT_TOKENS = 400
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05

# ─────────────────────────────────────────────
# TASK 1: Load data
# ─────────────────────────────────────────────
tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
segments = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))

seg_map = segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

print(f"Loaded {len(tickets)} tickets")

# ─────────────────────────────────────────────
# TASK 2: Input guardrails
# ─────────────────────────────────────────────
def check_input_guardrails(text):
    """
    Returns (status, tokens) where status is one of:
      'skipped_empty', 'flagged_long', or None (proceed)
    """
    if not text or text.strip() == "":
        return "skipped_empty", 0
    tokens = est_tokens(text)
    if tokens > MAX_INPUT_TOKENS:
        return "flagged_long", tokens
    return None, tokens

# ─────────────────────────────────────────────
# TASK 3: Redact PII
# ─────────────────────────────────────────────
EMAIL_RE  = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE  = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE  = re.compile(r"ORD-\d{6}")

def redact(text):
    """Apply three redaction patterns in order. Returns (redacted_text, counts)."""
    n_email  = len(EMAIL_RE.findall(text))
    text     = EMAIL_RE.sub("[EMAIL]", text)
    n_phone  = len(PHONE_RE.findall(text))
    text     = PHONE_RE.sub("[PHONE]", text)
    n_order  = len(ORDER_RE.findall(text))
    text     = ORDER_RE.sub("[ORDER_ID]", text)
    return text, {"email": n_email, "phone": n_phone, "order": n_order}

# Verify: after redaction, no PII remains
def verify_no_pii(text):
    return (not EMAIL_RE.search(text) and
            not PHONE_RE.search(text) and
            not ORDER_RE.search(text))

# ─────────────────────────────────────────────
# TASK 4: Prompts
# ─────────────────────────────────────────────
ZERO_SHOT_SYSTEM = """You are a CLASSIFY support ticket classifier for Instamart.
Classify each ticket into exactly one of: billing, delivery, technical, fraud.
Reply with ONLY valid JSON in the form: {"category": "...", "reason": "..."}
Do not include any explanation outside the JSON object."""

FEW_SHOT_SYSTEM = """You are a CLASSIFY support ticket classifier for Instamart.
Classify each ticket into exactly one of: billing, delivery, technical, fraud.
Reply with ONLY valid JSON in the form: {"category": "...", "reason": "..."}

EXAMPLES:
User: I was charged twice and I want a refund for my order.
Assistant: {"category": "billing", "reason": "duplicate charge and refund request"}

User: There is an unauthorized transaction on my account. I did not place this order. This is fraud.
Assistant: {"category": "fraud", "reason": "unauthorized transaction reported by customer"}

User: My order has not arrived and the rider is unreachable.
Assistant: {"category": "delivery", "reason": "late or missing delivery"}

User: The app crashes whenever I try to open my cart.
Assistant: {"category": "technical", "reason": "app crash reported"}"""

# ─────────────────────────────────────────────
# TASK 5: Resilient retry wrapper
# ─────────────────────────────────────────────
def call_with_retry(messages, ticket_id, max_retries=3):
    """Call mock_llm.chat with exponential backoff + jitter on TransientLLMError."""
    for attempt in range(max_retries + 1):
        try:
            return chat(messages)
        except TransientLLMError as e:
            if attempt == max_retries:
                raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            logger.warning(f"Ticket {ticket_id} | retry {attempt} | wait {wait:.4f}s | error: {e}")
            time.sleep(wait)

# ─────────────────────────────────────────────
# TASK 6: Output validation
# ─────────────────────────────────────────────
def validate_response(content):
    """
    Attempt to parse a category from the model response.
    Returns (category, status) where status ∈ {classified_json, extracted_text, fallback_technical}.
    """
    # Try JSON parse
    try:
        parsed = json.loads(content)
        cat = str(parsed.get("category", "")).lower().strip()
        if cat in ALLOWED_CATEGORIES:
            return cat, "classified_json"
        # JSON but bad category → extract from text
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Try regex extraction
    m = re.search(r"\b(billing|delivery|technical|fraud)\b", content or "", re.I)
    if m:
        return m.group(1).lower(), "extracted_text"

    # Fallback
    return "technical", "fallback_technical"

# ─────────────────────────────────────────────
# TASK: Full triage pipeline
# ─────────────────────────────────────────────
def run_triage(use_few_shot=True):
    """Run the full triage pipeline and return a DataFrame of results."""
    mock_llm.reset()

    system_prompt = FEW_SHOT_SYSTEM if use_few_shot else ZERO_SHOT_SYSTEM

    total_email = total_phone = total_order = 0
    results = []

    for _, row in tickets.iterrows():
        tid    = row["ticket_id"]
        cid    = row["customer_id"]
        text   = row["text"]
        true_cat = row["true_category"]

        # Guardrails
        guard_status, token_count = check_input_guardrails(text)

        if guard_status:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": guard_status,
                "predicted_category": None,
                "true_category": true_cat,
                "segment_name": seg_map.get(cid, {}).get("segment_name", "Unknown"),
                "is_premium": seg_map.get(cid, {}).get("is_premium", False),
                "priority": None,
                "tokens_used": 0,
                "prompt_type": None,
            })
            continue

        # Redact PII
        redacted, counts = redact(text)
        total_email += counts["email"]
        total_phone += counts["phone"]
        total_order += counts["order"]

        assert verify_no_pii(redacted), f"PII found after redaction in ticket {tid}!"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": redacted},
        ]

        resp = call_with_retry(messages, tid)
        content = resp.choices[0].message.content
        tokens_used = resp.usage.total_tokens

        category, classify_status = validate_response(content)

        # Priority routing
        seg_info = seg_map.get(cid, {})
        is_premium = seg_info.get("is_premium", False)
        if category == "fraud":
            priority = "high"
        elif is_premium:
            priority = "high"
        else:
            priority = "standard"

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": classify_status,
            "predicted_category": category,
            "true_category": true_cat,
            "segment_name": seg_info.get("segment_name", "Unknown"),
            "is_premium": is_premium,
            "priority": priority,
            "tokens_used": tokens_used,
            "prompt_type": "few_shot" if use_few_shot else "zero_shot",
        })

    print(f"\nRedaction totals: email={total_email}, phone={total_phone}, order={total_order}")
    print(f"Total substitutions: {total_email+total_phone+total_order}")
    print("Reference: 55 emails, 91 phones, 132 orders, 278 total")
    return pd.DataFrame(results), (total_email, total_phone, total_order)

print("\n" + "=" * 60)
print("Running triage pipeline (few-shot)…")
print("=" * 60)
df_results, pii_counts = run_triage(use_few_shot=True)
df_results.to_csv(os.path.join(OUTPUTS, "triage_results.csv"), index=False)
print(f"Saved {len(df_results)} rows to outputs/triage_results.csv")

# Summary statistics
status_counts = df_results["status"].value_counts()
print(f"\nStatus breakdown:\n{status_counts.to_string()}")
print(f"\nReference: 4 empty, 4 long, 156 classified (all recovered from TransientLLMError)")

# Accuracy on classifiable tickets
classifiable = df_results[df_results["status"].isin(["classified_json","extracted_text","fallback_technical"])]
classifiable = classifiable[classifiable["true_category"].isin(ALLOWED_CATEGORIES)]
if len(classifiable) > 0:
    acc = (classifiable["predicted_category"] == classifiable["true_category"]).mean()
    print(f"\nClassification accuracy (on labelled non-special tickets): {acc:.3f}")

priority_counts = df_results[df_results["priority"].notna()]["priority"].value_counts()
print(f"\nPriority breakdown:\n{priority_counts.to_string()}")

# ─────────────────────────────────────────────
# TASK: Token budget demo
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Token budget demo")
print("=" * 60)
BUDGET_PER_TICKET = 200  # tokens
budget_rows = []
mock_llm.reset()
system_prompt = FEW_SHOT_SYSTEM

for _, row in tickets.iterrows():
    tid  = row["ticket_id"]
    text = row["text"]
    guard_status, token_count = check_input_guardrails(text)
    if guard_status:
        budget_rows.append({
            "ticket_id": tid,
            "status": guard_status,
            "tokens_used": 0,
            "within_budget": True,
        })
        continue

    redacted, _ = redact(text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": redacted},
    ]
    est = est_tokens(redacted) + est_tokens(system_prompt)

    if est > BUDGET_PER_TICKET:
        budget_rows.append({
            "ticket_id": tid,
            "status": "budget_exceeded",
            "tokens_used": est,
            "within_budget": False,
        })
        continue

    try:
        resp = call_with_retry(messages, tid)
        budget_rows.append({
            "ticket_id": tid,
            "status": "classified",
            "tokens_used": resp.usage.total_tokens,
            "within_budget": resp.usage.total_tokens <= BUDGET_PER_TICKET,
        })
    except TransientLLMError:
        budget_rows.append({
            "ticket_id": tid,
            "status": "error",
            "tokens_used": 0,
            "within_budget": False,
        })

budget_df = pd.DataFrame(budget_rows)
budget_df.to_csv(os.path.join(OUTPUTS, "triage_budget_demo.csv"), index=False)
print(f"Budget demo results saved to outputs/triage_budget_demo.csv")
print(budget_df["status"].value_counts().to_string())

print("\n" + "=" * 60)
print("Part C complete.")
print("=" * 60)
