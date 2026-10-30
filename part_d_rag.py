"""
part_d_rag.py
-------------
Part D: Policy RAG Assistant and Refund Agent.

Sub-task 1: Policy RAG assistant — retrieves relevant policy text and answers questions.
Sub-task 2: Refund agent — processes agent_scenarios.csv, runs tool calls, persists state.
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import re
import sqlite3
import math

import pandas as pd

import mock_llm
from mock_llm import chat, est_tokens, TransientLLMError

OUTPUTS = "outputs"
STATE_DIR = "state"
POLICIES_DIR = os.path.join("data", "policies")
DB = os.path.join("data", "instamart.db")

os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# PART D.1 — POLICY RAG ASSISTANT
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Build policy index
# ─────────────────────────────────────────────
def load_policies(policies_dir):
    """Load all .txt policy files into a dict {filename: content}."""
    docs = {}
    for fn in sorted(os.listdir(policies_dir)):
        if fn.endswith(".txt"):
            with open(os.path.join(policies_dir, fn), encoding="utf-8") as f:
                docs[fn] = f.read().strip()
    return docs

STOP_WORDS = {
    "the","a","an","is","of","to","for","in","on","how","what","when","who","are","do","does",
    "i","my","be","can","after","from","it","or","and","at","by","which","why","if","than","that"
}

def stems(text):
    """Return set of 5-char stems for non-stopword words in text."""
    return {w[:5] for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP_WORDS}

def retrieve(question, docs, top_k=2):
    """
    TF-IDF-style retrieval: score each document by inverse-document-frequency
    weighted stem overlap with the question.
    Returns list of (filename, excerpt) for top_k docs.
    """
    q_stems = stems(question)
    # Compute document frequency for each stem
    df = {s: sum(s in stems(body) for body in docs.values()) for s in q_stems}

    scores = {}
    for fn, body in docs.items():
        doc_stems = stems(body)
        sc = sum(1.0 / df[s] for s in q_stems & doc_stems if df[s] > 0)
        scores[fn] = sc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Only return docs with a non-zero score (i.e., relevant ones)
    relevant = [(fn, docs[fn]) for fn, sc in ranked if sc > 0][:top_k]
    return relevant

def build_context(retrieved_docs):
    """Concatenate retrieved doc excerpts with filename tags."""
    parts = []
    for fn, body in retrieved_docs:
        parts.append(f"[{fn}]\n{body}")
    return "\n\n".join(parts)

POLICY_SYSTEM = """You are a POLICY ASSISTANT for Instamart customer support.
Answer questions using ONLY the provided policy context.
If the answer is not in the context, say: I could not find that information in the policy documents.
Be concise and factual."""

def answer_question(question, docs):
    """Retrieve relevant docs and ask the LLM to answer."""
    retrieved = retrieve(question, docs)
    if not retrieved:
        context = ""
    else:
        context = build_context(retrieved)

    user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    messages = [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    resp = chat(messages)
    return resp.choices[0].message.content, [fn for fn, _ in retrieved], resp.usage.total_tokens

# ─────────────────────────────────────────────
# Run policy evaluation
# ─────────────────────────────────────────────
print("=" * 60)
print("PART D.1: Policy RAG Evaluation")
print("=" * 60)

docs = load_policies(POLICIES_DIR)
print(f"Loaded {len(docs)} policy documents: {list(docs.keys())}")

policy_eval = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"))

eval_results = []
correct_doc = 0
correct_fact = 0
total_answerable = 0

for _, row in policy_eval.iterrows():
    q     = row["question"]
    e_doc = row["expected_doc"]
    e_fact = row["expected_fact"]

    answer, retrieved_docs, tokens = answer_question(q, docs)

    # Check doc retrieval
    doc_hit = (e_doc == "NONE" and not retrieved_docs) or (e_doc != "NONE" and e_doc in retrieved_docs)
    # Check fact presence (for answerable questions)
    fact_hit = False
    if e_doc != "NONE" and e_fact:
        total_answerable += 1
        fact_hit = e_fact.lower() in answer.lower()
        if fact_hit:
            correct_fact += 1
    if doc_hit:
        correct_doc += 1

    eval_results.append({
        "question": q,
        "expected_doc": e_doc,
        "expected_fact": e_fact,
        "retrieved_docs": "; ".join(retrieved_docs),
        "answer": answer,
        "doc_hit": doc_hit,
        "fact_hit": fact_hit,
        "tokens": tokens,
    })
    print(f"\nQ: {q}")
    print(f"   Expected doc: {e_doc}")
    print(f"   Retrieved:    {retrieved_docs}")
    print(f"   Answer:       {answer[:120]}")
    print(f"   Doc hit: {doc_hit} | Fact hit: {fact_hit}")

print(f"\nDocument retrieval accuracy: {correct_doc}/{len(policy_eval)} = {correct_doc/len(policy_eval)*100:.1f}%")
print(f"Fact extraction accuracy:    {correct_fact}/{total_answerable} = {correct_fact/total_answerable*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# PART D.2 — REFUND AGENT
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART D.2: Refund Agent")
print("=" * 60)

# ─────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────
conn = sqlite3.connect(DB)

def get_order(order_id: str) -> dict:
    """Look up order from the database."""
    row = pd.read_sql_query(
        "SELECT * FROM orders WHERE order_id = ?", conn, params=(order_id,)
    )
    if row.empty:
        return {"error": f"Order {order_id} not found"}
    r = row.iloc[0]
    return {
        "order_id":         r["order_id"],
        "customer_id":      r["customer_id"],
        "status":           r["status"],
        "order_amount":     float(r["order_amount"]),
        "delivery_minutes": None if pd.isna(r["delivery_minutes"]) else int(r["delivery_minutes"]),
    }

def search_policy(query: str) -> dict:
    """Retrieve relevant policy sentences."""
    retrieved = retrieve(query, docs, top_k=2)
    return {"policy": build_context(retrieved)}

def calculate_refund(order_amount: float, delivery_minutes) -> dict:
    """Apply refund policy: full refund if delivered > 45 min, else 0."""
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0, "reason": "Delivery was within 45 minutes or not applicable."}
    return {"refund_inr": round(float(order_amount), 2), "reason": "Delivery exceeded 45 minutes — full refund."}

def issue_refund(order_id: str, refund_inr: float) -> dict:
    """Issue a refund (mock)."""
    return {"order_id": order_id, "refund_inr": refund_inr, "status": "issued"}

def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    """Escalate to human for refunds above INR 1000 (mock)."""
    return {"order_id": order_id, "refund_inr": refund_inr, "status": "escalated"}

TOOL_MAP = {
    "get_order":        get_order,
    "search_policy":    search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund":     issue_refund,
    "escalate_to_human":escalate_to_human,
}

TOOLS_SPEC = [
    {"name": "get_order",         "description": "Look up order details by order_id"},
    {"name": "search_policy",     "description": "Search policy documents for a query"},
    {"name": "calculate_refund",  "description": "Calculate refund based on order_amount and delivery_minutes"},
    {"name": "issue_refund",      "description": "Issue a refund for an order"},
    {"name": "escalate_to_human", "description": "Escalate a high-value refund to a human lead"},
]

# ─────────────────────────────────────────────
# Agent loop with state persistence
# ─────────────────────────────────────────────
AGENT_SYSTEM = """You are an AGENT for Instamart customer support refund processing.
Use the provided tools to look up orders, check policy, calculate and issue refunds.
Follow the refund policy strictly."""

MAX_STEPS = 10  # safety limit — if exceeded → FALLBACK

def dispatch_tool(tool_name, arguments):
    """Call the appropriate tool function."""
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return fn(**arguments)
    except Exception as e:
        return {"error": str(e)}

def save_state(scenario_id, messages, step):
    """Persist conversation state to state/ directory."""
    path = os.path.join(STATE_DIR, f"{scenario_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenario_id": scenario_id, "step": step, "messages": messages}, f, indent=2, default=str)

def load_state(scenario_id):
    """Load persisted state, or return None if not found."""
    path = os.path.join(STATE_DIR, f"{scenario_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def run_agent(scenario_id, customer_id, user_message, resume=False):
    """
    Run the refund agent for a single scenario.
    Returns (outcome, refund_inr, tools_executed, steps).
    """
    mock_llm.reset()

    # Resume or start fresh
    if resume:
        state = load_state(scenario_id)
        if state:
            messages  = state["messages"]
            start_step = state["step"]
            print(f"  Resuming {scenario_id} from step {start_step}")
        else:
            messages  = [{"role": "system", "content": AGENT_SYSTEM},
                         {"role": "user",   "content": user_message}]
            start_step = 0
    else:
        messages  = [{"role": "system", "content": AGENT_SYSTEM},
                     {"role": "user",   "content": user_message}]
        start_step = 0

    tools_executed = []
    outcome = None
    refund_inr = 0.0

    for step in range(start_step, MAX_STEPS):
        save_state(scenario_id, messages, step)

        resp = chat(messages, tools=TOOLS_SPEC)
        msg  = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "tool_calls" and msg.tool_calls:
            # Execute each tool call
            for tc in msg.tool_calls:
                t_name = tc.name
                t_args = tc.arguments
                print(f"  Step {step}: tool_call → {t_name}({t_args})")
                result = dispatch_tool(t_name, t_args)
                print(f"             result:    {result}")
                tools_executed.append(t_name)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc.id, "name": t_name, "arguments": t_args}],
                })
                messages.append({
                    "role": "tool",
                    "name": t_name,
                    "content": json.dumps(result),
                })

                # Check for FALLBACK (loop detection)
                if tools_executed.count("search_policy") > 3:
                    outcome = "FALLBACK"
                    print(f"  FALLBACK: loop detected (search_policy called >3 times)")
                    break

            if outcome:
                break
        else:
            # Final text response
            content = msg.content or ""
            print(f"  Step {step}: final response → {content[:120]}")
            if content.startswith("REFUND:"):
                m = re.search(r"INR\s+([\d.]+)", content)
                refund_inr = float(m.group(1)) if m else 0.0
                outcome = "REFUND"
            elif content.startswith("ESCALATE:"):
                # Extract amount from tool results
                for m_msg in messages:
                    if m_msg.get("role") == "tool" and m_msg.get("name") == "calculate_refund":
                        data = json.loads(m_msg["content"])
                        refund_inr = data.get("refund_inr", 0.0)
                        break
                outcome = "ESCALATE"
            elif content.startswith("DENIED:"):
                outcome = "DENIED"
            elif content.startswith("NOT_FOUND:"):
                outcome = "NOT_FOUND"
            elif content.startswith("NEEDS_INFO:"):
                outcome = "NEEDS_INFO"
            else:
                outcome = "UNKNOWN"
            break

    if outcome is None:
        outcome = "FALLBACK"

    # Clean up state file after completion (unless FALLBACK for resume demo)
    return outcome, refund_inr, tools_executed, step + 1

# ─────────────────────────────────────────────
# Run all 7 scenarios
# ─────────────────────────────────────────────
scenarios = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
print(f"\nLoaded {len(scenarios)} agent scenarios\n")

agent_runs = []
refund_ledger = []

for _, row in scenarios.iterrows():
    sid      = row["scenario_id"]
    cid      = row["customer_id"]
    msg      = row["message"]
    expected = row["expected_outcome"]
    exp_refund = float(row["expected_refund_inr"])

    print(f"\n{'─'*60}")
    print(f"Scenario {sid}: {msg[:80]}")
    print(f"Expected outcome: {expected} | Expected refund: INR {exp_refund:.2f}")

    outcome, refund_inr, tools, steps = run_agent(sid, cid, msg, resume=False)

    match = (outcome == expected)
    print(f"Actual outcome:   {outcome} | Actual refund: INR {refund_inr:.2f} | Match: {match}")

    run_record = {
        "scenario_id":    sid,
        "customer_id":    cid,
        "message":        msg,
        "expected_outcome": expected,
        "actual_outcome": outcome,
        "expected_refund_inr": exp_refund,
        "actual_refund_inr":   refund_inr,
        "outcome_match":  match,
        "tools_executed": json.dumps(tools),
        "steps":          steps,
    }
    agent_runs.append(run_record)

    if outcome in ("REFUND", "ESCALATE"):
        refund_ledger.append({
            "scenario_id":   sid,
            "order_id":      re.search(r"ORD-\d{6}", msg).group(0) if re.search(r"ORD-\d{6}", msg) else "",
            "customer_id":   cid,
            "refund_inr":    refund_inr,
            "outcome":       outcome,
            "approved":      outcome == "REFUND",
            "escalated":     outcome == "ESCALATE",
        })

# Save outputs
runs_path   = os.path.join(OUTPUTS, "agent_runs.jsonl")
ledger_path = os.path.join(OUTPUTS, "refund_ledger.jsonl")

with open(runs_path, "w", encoding="utf-8") as f:
    for r in agent_runs:
        f.write(json.dumps(r) + "\n")

with open(ledger_path, "w", encoding="utf-8") as f:
    for r in refund_ledger:
        f.write(json.dumps(r) + "\n")

print(f"\n\nSaved {len(agent_runs)} agent runs to {runs_path}")
print(f"Saved {len(refund_ledger)} ledger entries to {ledger_path}")

# Results table
print("\n" + "=" * 60)
print("AGENT SCENARIO RESULTS TABLE")
print("=" * 60)
for r in agent_runs:
    print(f"  {r['scenario_id']:3s} | Expected: {r['expected_outcome']:10s} | Got: {r['actual_outcome']:10s} | "
          f"Refund: INR {r['actual_refund_inr']:>9.2f} | Match: {r['outcome_match']}")

# ─────────────────────────────────────────────
# STOP-SAFELY demo (scenario S7 = FALLBACK)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STOP-SAFELY DEMONSTRATION (FALLBACK scenario S7)")
print("=" * 60)
s7 = scenarios[scenarios["scenario_id"] == "S7"].iloc[0]
outcome7, _, tools7, steps7 = run_agent("S7_demo", s7["customer_id"], s7["message"], resume=False)
print(f"Stop-safely status: outcome={outcome7}, steps={steps7}, tools={tools7}")

# ─────────────────────────────────────────────
# CRASH-AND-RESUME demo (use S1)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CRASH-AND-RESUME DEMONSTRATION (scenario S1)")
print("=" * 60)
s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]

# Step 1: run until get_order returns (simulate crash after step 0)
mock_llm.reset()
messages_crash = [
    {"role": "system", "content": AGENT_SYSTEM},
    {"role": "user",   "content": s1["message"]},
]
# Simulate one step
resp1 = chat(messages_crash, tools=TOOLS_SPEC)
tc1   = resp1.choices[0].message.tool_calls[0]
result1 = dispatch_tool(tc1.name, tc1.arguments)
messages_crash.append({
    "role": "assistant", "content": None,
    "tool_calls": [{"id": tc1.id, "name": tc1.name, "arguments": tc1.arguments}],
})
messages_crash.append({"role": "tool", "name": tc1.name, "content": json.dumps(result1)})
tools_after_crash = [tc1.name]
save_state("S1_resume", messages_crash, 1)
print(f"Crash simulated after step 0. Tool executed: {tc1.name}")
print(f"State saved to state/S1_resume.json")

# Step 2: resume from saved state
print("Resuming from saved state…")
state_loaded = load_state("S1_resume")
messages_resume = state_loaded["messages"]
start_step = state_loaded["step"]

for step in range(start_step, MAX_STEPS):
    resp = chat(messages_resume, tools=TOOLS_SPEC)
    msg  = resp.choices[0].message
    finish = resp.choices[0].finish_reason
    if finish == "tool_calls" and msg.tool_calls:
        for tc in msg.tool_calls:
            result = dispatch_tool(tc.name, tc.arguments)
            tools_after_crash.append(tc.name)
            print(f"  Resume step {step}: {tc.name} → {result}")
            messages_resume.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}],
            })
            messages_resume.append({"role": "tool", "name": tc.name, "content": json.dumps(result)})
    else:
        content = msg.content or ""
        print(f"  Resume step {step}: final → {content[:100]}")
        break

print(f"\nFull tool sequence in crash-and-resume: {tools_after_crash}")

conn.close()

print("\n" + "=" * 60)
print("Part D complete.")
print("=" * 60)
