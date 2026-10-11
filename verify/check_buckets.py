# -*- coding: utf-8 -*-
"""Gate: the engine picks the right bucket and escalates when a request does not fit.

Runs with onnxruntime + numpy only. Loads the default bucket, then sends requests that are
too long for it: the engine must swap to the smallest larger bucket that exists and fits
(one decoder in memory at a time), give the same decision as the golden reference, and raise
the ValueError (naming the needed L) only when no existing bucket fits. An explicit
``graph=`` is never swapped.
"""
import argparse
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide import DEFAULT_SEQ_LEN, DecisionEngine  # noqa: E402
from intern_decide.runtime import find_buckets  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def padded(request, words):
    """The golden request with ``words`` filler words added to the state (more tokens, same questions)."""
    out = json.loads(json.dumps(request))
    out["state"] = out["state"] + " " + " ".join(["note"] * words)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    onnx_dir = ROOT / args.model / "onnx"
    buckets = list(find_buckets(onnx_dir))
    print("buckets on disk: %s" % buckets)
    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    case = golden["cases"][0]
    problems = 0

    def check(label, ok, detail=""):
        nonlocal problems
        problems += not ok
        print("%s %-52s %s" % ("ok " if ok else "XX ", label, detail))

    engine = DecisionEngine(ROOT / args.model, tokenizer_json=ROOT / "models/hf/tokenizer.json",
                            temperature=golden["temperature"], intra_op_threads=args.threads)
    first = min(b for b in buckets if b >= DEFAULT_SEQ_LEN)
    check("default loads the smallest bucket >= %d" % DEFAULT_SEQ_LEN, engine.seq_len == first,
          "L=%d" % engine.seq_len)
    base = engine.predict(case["request"])
    check("a request that fits stays on the loaded bucket", engine.seq_len == first)
    reference = {k: v["decision"] for k, v in case["answers"].items()}
    check("decisions equal the golden reference",
          {k: v["decision"] for k, v in base["answers"].items()} == reference)

    # grow the prompt past each bucket in turn
    for target in [b for b in buckets if b > engine.seq_len]:
        previous = engine.seq_len
        words = int(target * 0.95) - base["usage"]["input_tokens"]   # lands in (previous, target]
        request = padded(case["request"], max(words, 1))
        result = engine.predict(request)
        tokens = result["usage"]["input_tokens"]
        expected = min(b for b in buckets if b >= tokens)
        check("%d tokens: swap L=%d -> %d" % (tokens, previous, engine.seq_len),
              engine.seq_len == expected and engine.seq_len > previous,
              "expected L=%d" % expected)
        same = {k: v["decision"] for k, v in result["answers"].items()} == reference
        print("   (informational) decisions %s the short prompt's: filler text may legitimately move an answer"
              % ("equal" if same else "differ from"))

    # nothing fits: the ValueError names the needed L
    too_long = padded(case["request"], buckets[-1] + 300)
    try:
        engine.predict(too_long)
        check("no bucket fits -> ValueError", False, "no exception")
    except ValueError as exc:
        check("no bucket fits -> ValueError names the needed L", "--seq-len" in str(exc), str(exc)[:90])

    del engine                       # one decoder in memory at a time
    gc.collect()
    # an explicit graph= is used as it is and never swapped
    pinned = DecisionEngine(ROOT / args.model, graph="model_L%d.onnx" % first,
                            tokenizer_json=ROOT / "models/hf/tokenizer.json",
                            temperature=golden["temperature"], intra_op_threads=args.threads)
    check("graph= wins over seq_len", pinned.seq_len == first)
    try:
        pinned.predict(padded(case["request"], 450))
        check("graph= is not swapped (ValueError)", len(buckets) == 1, "no exception")
    except ValueError:
        check("graph= is not swapped (ValueError)", pinned.seq_len == first)

    print("\nproblems: %d" % problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
