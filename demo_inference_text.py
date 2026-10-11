# -*- coding: utf-8 -*-
"""Text decisions with Intern-Decision-0.8B, on onnxruntime + numpy.

    python demo_inference_text.py
    python demo_inference_text.py --state "..." --choice "team=billing,technical"

Every question is answered in ONE forward pass: the assistant turn is a JSON
skeleton whose values are <decision> markers, and the logits just before each
marker pick that field's answer symbol. Nothing is generated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from intern_decide import DEFAULT_SEQ_LEN, DecisionEngine, gpu_providers  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE = {
    "state": {
        "channel": "support email",
        "body": "I was charged twice for my Pro subscription this month. "
                "This is the second time it has happened and nobody replied to my last mail.",
    },
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
        "angry": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {}},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high", "critical"],
        },
    },
}


def parse_questions(args) -> dict:
    questions: dict = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        values = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = {
            "type": "choice", "instructions": args.instructions or name.strip(),
            "criteria": {v: v for v in values},
        }
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = {
            "type": "score", "instructions": args.instructions or name.strip(),
            "criteria": [v.strip() for v in rest.split(",") if v.strip()],
        }
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = {
            "type": "noul", "instructions": rest.strip() or name.strip(), "criteria": {},
        }
    return questions


def show(result: dict) -> None:
    for field, answer in result["answers"].items():
        spread = "  ".join(
            "%s=%.3f" % kv
            for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
        )
        extra = ""
        if answer["type"] == "score":
            extra = "  (expected %.2f)" % answer["score"]
        elif answer["type"] == "noul":
            extra = "  (p(yes)=%.3f)" % answer["noul"]
        print("%-10s %-22s %.4f%s\n           %s"
              % (field, answer["decision"], answer["confidence"], extra, spread))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="text or JSON; omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=question text")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default="models/onnx")
    parser.add_argument("--tokenizer", default="models/hf/tokenizer.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="sequence-length bucket to load (model_L<N>.onnx; default: %d). A longer request "
                             "loads the next larger bucket that exists" % DEFAULT_SEQ_LEN)
    parser.add_argument("--gpu", action="store_true",
                        help="use CUDA / DirectML if this onnxruntime build has it "
                             "(needs onnxruntime-gpu or onnxruntime-directml); falls back to CPU")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.state is None:
        request = EXAMPLE
    else:
        try:
            state = json.loads(args.state)
        except json.JSONDecodeError:
            state = args.state
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")
        request = {"state": state, "questions": questions}

    started = time.perf_counter()
    providers = gpu_providers() if args.gpu else None
    if args.gpu and providers[0] == "CPUExecutionProvider":
        print("warning: no GPU provider in this onnxruntime build; running on CPU "
              "(install onnxruntime-gpu or onnxruntime-directml instead of onnxruntime)", file=sys.stderr)
    engine = DecisionEngine(ROOT / args.model, tokenizer_json=ROOT / args.tokenizer,
                            seq_len=args.seq_len, intra_op_threads=args.threads, providers=providers)
    load = time.perf_counter() - started
    if args.gpu and providers[0] != "CPUExecutionProvider" and engine.provider == "CPUExecutionProvider":
        print("warning: %s failed to load; fell back to CPU (check that the CUDA 13 / cuDNN 9 folders are on PATH)"
              % providers[0], file=sys.stderr)

    started = time.perf_counter()
    result = engine.predict(request)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("state:")
    print("  %s\n" % json.dumps(request["state"], ensure_ascii=False)[:160])
    show(result)
    print("\n%d prompt tokens, %d decisions in one forward pass"
          % (result["usage"]["input_tokens"], result["usage"]["decision_count"]))
    print("load %.1f s, inference %.2f s (graph fixed at %d tokens, %s)"
          % (load, elapsed, engine.seq_len, engine.provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
