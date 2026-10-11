# -*- coding: utf-8 -*-
"""Put a runnable model under models/.

Every repository in this family exposes the same entry point; what it does
differs. Intern-Decision has no published ONNX, so this downloads the original
checkpoint over plain HTTPS and exports it here.

    python export/prepare_model.py --list
    python export/prepare_model.py                  # default: L=384
    python export/prepare_model.py --seq-len 1024   # more questions per request (a second bucket; shares the weights)

No git-lfs. The checkpoint comes from Hugging Face's resolve endpoint using only
the standard library; the export itself needs requirements-export.txt.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

CHECKPOINT = "internlm/Intern-Decision-0.8B"
REVISION = "main"
HF_DIR = ROOT / "models" / "hf"
ONNX_DIR = ROOT / "models" / "onnx"

#: What the export reads. The three safetensors shards are the bulk (~1.7 GB).
CHECKPOINT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
    "model-language-0001-save_rank0.safetensors",
    "model-vision-0001-save_rank0.safetensors",
    "model-projector-0001-save_rank0.safetensors",
    "inference.py",
]

#: The graph is traced at one fixed prompt length -- see README and
#: export_onnx.py for why it cannot be dynamic. Cost scales with L, so pick the
#: smallest that fits the requests you actually send.
BUCKETS = {
    384: "the default: one question with a 224 px picture (~260-375 tokens) fits",
    512: "~6 questions per request",
    1024: "~13 questions per request",
    1280: "the protocol maximum of 16 questions",
}


def run(argv: list[str]) -> None:
    print("$ %s" % " ".join(str(a) for a in argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT)


def fetch_checkpoint() -> None:
    print("checkpoint -> %s" % HF_DIR, flush=True)
    fetch_files(CHECKPOINT, CHECKPOINT_FILES, HF_DIR, revision=REVISION)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the sequence-length buckets")
    parser.add_argument("--seq-len", type=int, default=384,
                        help="fixed prompt length to trace the graph at")
    parser.add_argument("--skip-vision", action="store_true",
                        help="text-only build: do not export the vision encoder")
    parser.add_argument("--skip-golden", action="store_true",
                        help="do not regenerate artifacts/golden.json")
    parser.add_argument("--fetch-only", action="store_true",
                        help="download the checkpoint and stop")
    parser.add_argument("--no-shard", action="store_true",
                        help="leave the weights in one file, over GitHub's 2 GiB asset limit")
    parser.add_argument("--shard-limit-mb", type=int, default=1900,
                        help="maximum size of one weight file (default: 1900)")
    args = parser.parse_args()

    if args.list:
        print("sequence-length buckets (cost scales with L, not with the real prompt):\n")
        for length, note in BUCKETS.items():
            print("  --seq-len %-5d %s" % (length, note))
        print("\nAny multiple of 64 works. The runtime right-pads up to L and")
        print("raises if a request is longer, naming the L you would need.")
        print("\nThe checkpoint is ~1.7 GB; the exported graph is ~3.1 GB per bucket.")
        return 0

    fetch_checkpoint()
    if args.fetch_only:
        return 0

    if not args.skip_golden:
        run([sys.executable, "export/make_golden.py"])
        if not args.skip_vision:
            run([sys.executable, "export/make_image_golden.py"])

    export = [sys.executable, "export/export_onnx.py", "--seq-len", str(args.seq_len)]
    if args.skip_vision:
        export.append("--skip-vision")
    run(export)

    # fp32 weights land at ~3.0 GB in one blob, over GitHub's 2 GiB per-asset
    # limit. Splitting them changes no numbers, so do it unconditionally.
    # The weights do not depend on L, so every bucket reads one shared pool
    # (weights.onnx.data*): the first bucket's shards become the pool, later buckets
    # are redirected to it (identical bytes only) and keep just their graph.
    if not args.no_shard:
        graph = ONNX_DIR / "onnx" / ("model_L%d.onnx" % args.seq_len)
        pool = ONNX_DIR / "onnx" / "weights.onnx.data"
        if pool.exists():
            run([sys.executable, "export/share_external_data.py", str(graph)])
        else:
            run([sys.executable, "export/shard_external_data.py", str(graph),
                 "--limit-mb", str(args.shard_limit_mb)])
            run([sys.executable, "export/share_external_data.py", "--adopt", str(graph)])

    print("\nready: %s" % ONNX_DIR)
    print("verify with:  python verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
