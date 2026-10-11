# -*- coding: utf-8 -*-
"""Which sequence lengths does the exported graph actually get right?

The linear-attention layers chunk the sequence by 64 and pad up to a multiple of
it. If the exporter froze that padding at the traced length, only some lengths
will match torch. This walks lengths around the traced one and reports the
per-length logit error, so the failure is characterised rather than guessed at.

Run in .venv-ref (needs both torch and onnxruntime).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models/hf"
sys.path.insert(0, str(ROOT))

CHUNK = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx/onnx/model.onnx")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import onnxruntime as ort
    from transformers import Qwen3_5ForConditionalGeneration

    from export.export_onnx import DecisionGraph  # noqa: E402

    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    # Longest case gives us a token stream to slice to any length.
    stream = max((c["input_ids"] for c in golden["cases"]), key=len)

    print("loading torch model ...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(HF), dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    graph = DecisionGraph(model).eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(ROOT / args.model), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    traced = len(golden["cases"][0]["input_ids"])
    print("traced length was %d  (-> %d chunks of %d)\n"
          % (traced, -(-traced // CHUNK), CHUNK))

    for length in (128, 192, 200, 221, 244, 253, 255, 256, 257, 260, 271, 300, 312, 320, 384):
        ids = torch.tensor([stream[:length]], dtype=torch.long)
        if ids.shape[1] < length:
            continue
        mask = torch.ones_like(ids)
        positions = (
            torch.arange(length, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1).contiguous()
        )
        reads = torch.tensor([length - 5, length - 2], dtype=torch.long)

        with torch.no_grad():
            expected = graph(ids, mask, positions, reads)[0].numpy()
        got = session.run(["logits"], {
            "input_ids": ids.numpy(),
            "attention_mask": mask.numpy(),
            "position_ids": positions.numpy(),
            "read_positions": reads.numpy(),
        })[0][0]

        delta = float(np.abs(got - expected).max())
        chunks = -(-length // CHUNK)
        print("  len=%4d  chunks=%d  pad=%2d  max|dlogit|=%.3e  %s"
              % (length, chunks, (CHUNK - length % CHUNK) % CHUNK, delta,
                 "OK" if delta < 1e-2 else "WRONG"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
