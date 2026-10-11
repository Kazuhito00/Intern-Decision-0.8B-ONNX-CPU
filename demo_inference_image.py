# -*- coding: utf-8 -*-
"""Image decisions with Intern-Decision-0.8B, on onnxruntime + numpy + Pillow.

    python demo_inference_image.py                        # built-in example: artifacts/images/sample_chart.png (a bar chart)
    python demo_inference_image.py --image shot.png \
        --choice "status=paid,pending,failed"

Images ride in the same request as the questions and are answered in the same
single forward pass. Up to 8 images; each is encoded on its own (see README).

Every image is scaled down to 224 px on its long edge before it is given to the model (``--size``; 0 = as it is). The graph
is fixed at L=512 tokens: a 448 px picture already needs 570-600 tokens and does not fit, a 224 px one needs about 64.
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
SAMPLES = ROOT / "artifacts" / "images"


def built_in_example():
    """The sample picture: a bar chart drawn by wiki_qa/synthetic/gen.py (the synthetic set of the Wikipedia image benchmark).

    artifacts/images/sample_chart.png is syn-0001 of that set; the tallest bar is B. The pictures of that set are drawn by a script,
    so there is no photograph, no copyright question and nothing from Wikipedia in it.
    """
    return {
        "state": "Look at the attached image.",
        "images": [str(SAMPLES / "sample_chart.png")],
        "questions": {
            "tallest": {
                "type": "choice",
                "instructions": "Which bar is the tallest?",
                "criteria": {"A": "bar A", "B": "bar B", "C": "bar C", "D": "bar D"},
            },
            "chart": {
                "type": "noul",
                "instructions": "Is this picture a bar chart?",
                "criteria": {},
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


def scaled(path, size: int):
    """The picture, shrunk so that its long edge is at most ``size`` px (a Pillow image: the engine takes it as it is)."""
    from PIL import Image

    picture = Image.open(pathlib.Path(str(path)).expanduser()).convert("RGB")
    picture.thumbnail((size, size), Image.Resampling.LANCZOS)
    return picture


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", default=[], metavar="PATH",
                        help="attach an image (repeatable, up to 8)")
    parser.add_argument("--state", default="Look at the attached image.")
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
    parser.add_argument("--size", type=int, default=224,
                        help="long edge in pixels of the picture given to the model (default: 224; 0 = no resize)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.image:
        request = built_in_example()
    else:
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")
        request = {"state": args.state, "images": list(args.image), "questions": questions}

    names = [pathlib.Path(str(p)).name for p in request["images"]]
    if args.size:                                      # scale down (never up) to the long edge; the aspect ratio is kept
        request["images"] = [scaled(p, args.size) for p in request["images"]]

    started = time.perf_counter()
    providers = gpu_providers() if args.gpu else None
    if args.gpu and providers[0] == "CPUExecutionProvider":
        print("warning: no GPU provider in this onnxruntime build; running on CPU "
              "(install onnxruntime-gpu or onnxruntime-directml instead of onnxruntime)", file=sys.stderr)
    engine = DecisionEngine(ROOT / args.model, tokenizer_json=ROOT / args.tokenizer,
                            seq_len=args.seq_len, intra_op_threads=args.threads, media_root=pathlib.Path.cwd(), providers=providers)
    load = time.perf_counter() - started
    if args.gpu and providers[0] != "CPUExecutionProvider" and engine.provider == "CPUExecutionProvider":
        print("warning: %s failed to load; fell back to CPU (check that the CUDA 13 / cuDNN 9 folders are on PATH)"
              % providers[0], file=sys.stderr)
    if engine.vision is None:
        print("this build has no vision encoder; re-run "
              "`python export/prepare_model.py` without --skip-vision")
        return 1

    started = time.perf_counter()
    result = engine.predict(request)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("images: %s%s\n" % (", ".join(names), " (long edge %d px)" % args.size if args.size else ""))
    for field, answer in result["answers"].items():
        spread = "  ".join(
            "%s=%.3f" % kv
            for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
        )
        print("%-10s %-22s %.4f\n           %s"
              % (field, answer["decision"], answer["confidence"], spread))
    print("\n%d prompt tokens, %d decisions in one forward pass"
          % (result["usage"]["input_tokens"], result["usage"]["decision_count"]))
    print("load %.1f s, inference %.2f s" % (load, elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
