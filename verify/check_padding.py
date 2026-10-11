# -*- coding: utf-8 -*-
"""Does right-padding change the logits the protocol reads?

The fixed-length graph forces every request to be padded up to L. That is only
sound if padding after the read positions leaves them untouched -- which
causality says it should, but the linear-attention scan is unusual enough to be
worth measuring rather than assuming.

Run in .venv-ref (torch only; no ONNX needed).
"""
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models/hf"
sys.path.insert(0, str(ROOT))


def main():
    from transformers import Qwen3_5ForConditionalGeneration

    from export.export_onnx import DecisionGraph, padded_inputs

    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    pad_id = 248044  # <|endoftext|>

    print("loading torch model ...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(HF), dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    graph = DecisionGraph(model).eval()

    hidden = model.config.text_config.hidden_size
    empty = torch.zeros(0, hidden, dtype=torch.float32)
    no_slots = torch.zeros(0, dtype=torch.long)
    worst = 0.0
    for case in golden["cases"][:3]:
        ids = case["input_ids"]
        reads = torch.tensor(case["read_positions"], dtype=torch.long)

        # unpadded, exactly as inference.py runs it
        bare = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(bare)
        positions = (
            torch.arange(len(ids), dtype=torch.long).view(1, 1, -1).expand(3, 1, -1).contiguous()
        )
        with torch.no_grad():
            reference = graph(bare, mask, positions, reads, empty, no_slots)[0].numpy()

        for seq_len in (384, 512, 1024, 1280):
            if len(ids) > seq_len:
                continue
            p_ids, p_mask, p_pos = padded_inputs(ids, seq_len, pad_id)
            with torch.no_grad():
                padded = graph(p_ids, p_mask, p_pos, reads, empty, no_slots)[0].numpy()
            delta = float(np.abs(padded - reference).max())
            worst = max(worst, delta)
            print("  %-16s %3d tokens -> L=%-4d  max|dlogit| = %.3e  %s"
                  % (case["id"], len(ids), seq_len, delta,
                     "OK" if delta < 1e-3 else "CHANGES THE ANSWER"))

    print("\nworst padding-induced logit change: %.3e" % worst)
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
