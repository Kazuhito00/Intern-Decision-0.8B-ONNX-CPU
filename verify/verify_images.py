# -*- coding: utf-8 -*-
"""End-to-end image gate: ONNX answers vs the checkpoint's own inference.py.

Needs onnxruntime + numpy + Pillow (Pillow only because images are decoded).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide import DEFAULT_SEQ_LEN, DecisionEngine  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="bucket to load (default: %d)" % DEFAULT_SEQ_LEN)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    golden = json.loads((ROOT / "artifacts/golden_images.json").read_text(encoding="utf-8"))
    engine = DecisionEngine(
        ROOT / args.model, graph=args.graph, seq_len=args.seq_len,
        tokenizer_json=ROOT / "models/hf/tokenizer.json",
        temperature=golden["temperature"], intra_op_threads=args.threads,
        media_root=ROOT / "artifacts/images",
    )
    if engine.vision is None:
        print("no vision graph in this build")
        return 1

    failures = decided = total = 0
    ran_at = {}
    worst = 0.0
    for case in golden["cases"]:
        result = engine.predict(case["request"])
        ran_at[case["id"]] = engine.seq_len
        ok_tokens = result["usage"]["input_tokens"] == case["input_tokens"]
        bad = not ok_tokens
        print("%s %-14s %s  tokens=%s"
              % ("ok " if ok_tokens else "XX ", case["id"], case["image"],
                 "OK" if ok_tokens else "ours=%d ref=%d"
                 % (result["usage"]["input_tokens"], case["input_tokens"])))
        for field, expected in case["answers"].items():
            got = result["answers"][field]
            total += 1
            same = got["decision"] == expected["decision"]
            decided += same
            delta = max(abs(got["probabilities"][k] - expected["probabilities"][k])
                        for k in expected["probabilities"])
            worst = max(worst, delta)
            bad |= not same
            print("      %-8s %s ours=%-10s ref=%-10s max|dp|=%.2e"
                  % (field, "ok " if same else "XX ", got["decision"],
                     expected["decision"], delta))
        failures += bad

    print("\ndecisions %d/%d   worst |dp| %.3e   failures %d/%d"
          % (decided, total, worst, failures, len(golden["cases"])))
    counts = {}
    for length in ran_at.values():
        counts[length] = counts.get(length, 0) + 1
    print("cases run at L=%s   (requested L=%d)"
          % (", ".join("%d: %d" % kv for kv in sorted(counts.items())), args.seq_len))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
