# -*- coding: utf-8 -*-
"""Gate 4-5: ONNX logits and final answers vs the checkpoint's own inference.py.

Runs with onnxruntime + numpy only. golden.json holds the reference answers and
torch_logits.npz the reference logits at the read positions, both produced by
verify/make_golden.py / export_onnx.py in the torch environment.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide import DEFAULT_SEQ_LEN, DecisionEngine  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--graph", default=None, help="override graph_meta.json")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="bucket to load (default: %d)" % DEFAULT_SEQ_LEN)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    logits_path = ROOT / "artifacts/torch_logits.npz"
    torch_logits = np.load(logits_path) if logits_path.exists() else None

    engine = DecisionEngine(
        ROOT / args.model,
        graph=args.graph, seq_len=args.seq_len,
        tokenizer_json=ROOT / "models/hf/tokenizer.json",
        temperature=golden["temperature"],
        intra_op_threads=args.threads,
    )

    worst_logit = 0.0
    worst_prob = 0.0
    decisions_ok = fields_total = 0
    failures = 0
    ran_at = {}     # bucket each case actually ran at (a long case makes the engine load a bigger one)

    for index, case in enumerate(golden["cases"]):
        result = engine.predict(case["request"])
        ran_at[case["id"]] = engine.seq_len

        if torch_logits is not None:
            compiled, ids, reads, prepared = engine.encode(case["request"])
            ours = engine.logits(ids, reads, prepared)
            theirs = torch_logits["case%d" % index]
            worst_logit = max(worst_logit, float(np.abs(ours - theirs).max()))

        ok_tokens = result["usage"]["input_tokens"] == case["input_tokens"]
        case_bad = not ok_tokens
        lines = []
        for field, expected in case["answers"].items():
            got = result["answers"][field]
            fields_total += 1
            same = got["decision"] == expected["decision"]
            decisions_ok += same
            delta = max(
                abs(got["probabilities"][k] - expected["probabilities"][k])
                for k in expected["probabilities"]
            )
            worst_prob = max(worst_prob, delta)
            case_bad |= not same
            lines.append(
                "      %-10s %s ours=%-18s ref=%-18s max|dp|=%.2e"
                % (field, "ok " if same else "XX ", got["decision"], expected["decision"], delta)
            )
        failures += case_bad
        print("%s %-16s tokens=%s" % ("ok " if not case_bad else "XX ", case["id"],
                                      "OK" if ok_tokens else "DIFF"))
        for line in lines:
            print(line)

    print("\ndecisions %d/%d   worst |dp| %.3e" % (decisions_ok, fields_total, worst_prob))
    if torch_logits is not None:
        print("worst |dlogit| vs torch: %.3e" % worst_logit)
    counts = {}
    for length in ran_at.values():
        counts[length] = counts.get(length, 0) + 1
    print("cases run at L=%s   (requested L=%d)"
          % (", ".join("%d: %d" % kv for kv in sorted(counts.items())), args.seq_len))
    print("failures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
