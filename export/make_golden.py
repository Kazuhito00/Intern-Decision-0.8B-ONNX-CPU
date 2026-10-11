# -*- coding: utf-8 -*-
"""Golden outputs from the checkpoint's own inference.py (torch, CPU).

Runs in .venv-ref. Everything downstream is checked against these, so the file
must be readable without torch: JSON and .npy only, never a pickle.
"""
import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models/hf"
sys.path.insert(0, str(HF))

CASES = [
    {
        "id": "support_routing",
        "state": "Hi, I was charged twice for my Pro subscription this month, please fix.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "charges, refunds, invoices",
                    "technical": "bugs, outages, errors",
                    "sales": "pricing, upgrades, new contracts",
                },
            },
            "angry": {
                "type": "noul",
                "instructions": "Is the customer angry?",
                "criteria": {},
            },
        },
    },
    {
        "id": "incident",
        "state": "URGENT: production database is down, all customers affected, CEO asking for updates every 5 minutes.",
        "questions": {
            "priority": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high", "critical"],
            },
            "security": {"type": "noul", "instructions": "Is this a security incident?", "criteria": {}},
            "pager": {"type": "noul", "instructions": "Should the on-call engineer be paged?", "criteria": {}},
        },
    },
    {
        "id": "bug_triage",
        "state": {
            "title": "Export button crashes in Safari",
            "body": "Works in Chrome. Not blocking, we told users to switch browsers.",
            "reporter": "qa",
        },
        "questions": {
            "severity": {
                "type": "choice",
                "instructions": "How severe is this bug?",
                "criteria": {
                    "cosmetic": "cosmetic only, nothing is broken",
                    "degraded": "degraded but there is a workaround",
                    "blocking": "blocking, no workaround exists",
                },
            },
        },
    },
    {
        "id": "many_options",
        "state": "Can you send me the invoice for March? Also my login stopped working on the mobile app.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is the primary intent?",
                "criteria": {
                    "order_status": "asking where an order is",
                    "cancel_subscription": "wants to cancel",
                    "login_problem": "cannot sign in",
                    "bug_report": "reporting a defect",
                    "invoice_request": "asking for an invoice or receipt",
                    "other": "anything else",
                },
            },
            "requests": {
                "type": "score",
                "instructions": "How many separate requests are in this message?",
                "criteria": ["one", "two", "three or more"],
            },
        },
    },
    {
        "id": "japanese_state",
        "state": "先週の納品が遅れた件、まだ連絡がありません。至急対応してください。",
        "questions": {
            "angry": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {}},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high"],
            },
        },
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/golden.json")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    import torch  # noqa: F401  (imported for its side effect on transformers)
    from inference import DECISION_TOKEN, DecisionEngine, compile_row

    engine = DecisionEngine(str(HF), device="cpu", dtype=args.dtype, attn_implementation="eager")
    tokenizer = engine.tokenizer
    marker_id = engine.backend.marker_id
    print("marker <decision> id:", marker_id, "temperature:", engine.temperature)

    records = []
    for case in CASES:
        row = {k: v for k, v in case.items() if k != "id"}
        compiled = compile_row(row)
        _, batch, positions = engine.backend.encode(row)
        text = tokenizer.apply_chat_template(
            compiled.messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True,
        )
        result = engine.predict(row)
        records.append({
            "id": case["id"],
            "request": row,
            "prompt": text,
            "input_ids": batch["input_ids"][0].tolist(),
            "read_positions": positions.tolist(),
            "fields": list(compiled.fields),
            "symbols": {k: list(v) for k, v in compiled.symbols.items()},
            "answers": result["answers"],
            "input_tokens": result["usage"]["input_tokens"],
        })
        print("  %-16s %4d tokens, %d fields -> %s"
              % (case["id"], result["usage"]["input_tokens"], len(compiled.fields),
                 {k: v["decision"] for k, v in result["answers"].items()}))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "Intern-Decision-0.8B",
        "dtype": args.dtype,
        "temperature": engine.temperature,
        "decision_token": DECISION_TOKEN,
        "decision_token_id": marker_id,
        "cases": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
