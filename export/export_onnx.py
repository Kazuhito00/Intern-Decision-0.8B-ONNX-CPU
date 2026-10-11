# -*- coding: utf-8 -*-
"""Export Intern-Decision-0.8B's decision path to a single ONNX graph.

Only one prefill ever happens (``use_cache=False``), so no KV/recurrent state
crosses the graph boundary and the export stays a plain feed-forward graph::

    input_ids [B, L], attention_mask [B, L], position_ids [3, B, L],
    read_positions [K]                       ->  logits [B, K, vocab]

Two decisions are load-bearing:

**The gather at read_positions is baked in.** Without it the graph would return
[B, L, 248320] -- 500 MB per call at 512 tokens -- and waste the vocab
projection on positions nobody reads.

**The sequence length L is fixed, not dynamic.** The linear-attention layers run
``for i in range(num_chunks)`` over 64-token chunks (see
``transformers/models/qwen3_5/modeling_qwen3_5.py::chunk_gated_delta_rule``).
That is a Python loop, so the exporter unrolls it at the traced length. A graph
traced at 253 tokens silently ignores everything past chunk 4 on longer inputs
and raises an out-of-bounds Gather on shorter ones. Declaring L static makes ORT
reject a wrong length loudly instead; callers right-pad up to L, which is safe
because every read position precedes the padding and the model is causal.

Run in .venv-ref (torch + transformers + onnx + onnxscript).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models/hf"
sys.path.insert(0, str(ROOT))

CHUNK = 64  # chunk_gated_delta_rule's chunk_size


class DecisionGraph(nn.Module):
    """Prefill once, return only the logits the protocol reads.

    ``image_features`` / ``image_slots`` let the vision encoder's output be
    scattered into the token embeddings without a second decoder graph. For a
    text-only request both are empty, and the scatter is a no-op.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, read_positions,
                image_features, image_slots):
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.index_copy(
            1, image_slots, image_features.unsqueeze(0).to(embeds.dtype)
        )
        return self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=read_positions,
        ).logits


class VisionGraph(nn.Module):
    """pixel_values + image_grid_thw -> one embedding row per visual token."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, image_grid_thw):
        # One image per call: see export/vision_export_patch.py for why the
        # per-image splits can be dropped (and why they must be, to export).
        return self.model.get_image_features_single(pixel_values, image_grid_thw)


def padded_inputs(input_ids, seq_len: int, pad_id: int):
    """Right-pad to exactly seq_len; mask and positions follow the real tokens."""
    real = len(input_ids)
    if real > seq_len:
        raise ValueError("%d tokens exceeds the graph's fixed length %d" % (real, seq_len))
    ids = list(input_ids) + [pad_id] * (seq_len - real)
    mask = [1] * real + [0] * (seq_len - real)
    positions = np.arange(seq_len, dtype=np.int64)[None, None, :].repeat(3, axis=0)
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


DEFAULT_SEQ_LEN = 384   # the bucket the runtime loads unless asked otherwise


def write_meta(out_dir: pathlib.Path, pad_id: int, dtype: str) -> None:
    """graph_meta.json lists every bucket on disk; the default is L=384 when it exists.

    Exporting one more bucket must not repoint the default, so the file is rebuilt
    from what is in onnx/ rather than from the bucket just written.
    """
    import re

    found = sorted(int(m.group(1)) for p in (out_dir / "onnx").glob("model_L*.onnx")
                   if (m := re.fullmatch(r"model_L(\d+)\.onnx", p.name)))
    default = DEFAULT_SEQ_LEN if DEFAULT_SEQ_LEN in found else found[0]
    vision = out_dir / "onnx" / "vision_encoder.onnx"
    meta = {"seq_len": default, "pad_token_id": pad_id, "dtype": dtype,
            "chunk_size": CHUNK, "graph": "model_L%d.onnx" % default,
            "buckets": found,
            "vision_graph": vision.name if vision.exists() else None}
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/onnx")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="fixed prompt length the graph is traced at (multiple of 64)")
    ap.add_argument("--mode", default="dynamo", choices=["dynamo", "legacy"])
    ap.add_argument("--optimize", action="store_true",
                    help="run onnxscript graph optimization (very slow on 24 layers)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--skip-vision", action="store_true",
                    help="text-only build: do not export the vision encoder")
    args = ap.parse_args()

    if args.seq_len % CHUNK:
        raise SystemExit("--seq-len must be a multiple of %d" % CHUNK)

    from transformers import Qwen3_5ForConditionalGeneration

    out_dir = ROOT / args.out
    (out_dir / "onnx").mkdir(parents=True, exist_ok=True)
    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    pad_id = golden.get("pad_token_id", 248044)  # <|endoftext|>

    print("loading checkpoint (%s) ..." % args.dtype, flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(HF), dtype=getattr(torch, args.dtype), local_files_only=True,
        attn_implementation="eager",
    ).eval()
    graph = DecisionGraph(model).eval()

    case = golden["cases"][0]
    ids, mask, positions = padded_inputs(case["input_ids"], args.seq_len, pad_id)
    reads = torch.tensor(case["read_positions"], dtype=torch.long)
    # Trace with a non-empty image scatter so the branch is real; both the slot
    # count and the read count stay dynamic.
    hidden = model.config.text_config.hidden_size
    feats = torch.zeros(4, hidden, dtype=next(graph.parameters()).dtype)
    slots = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    with torch.no_grad():
        reference = graph(ids, mask, positions, reads, feats, slots)
    print("torch logits:", tuple(reference.shape),
          "(traced at L=%d, %d chunks)" % (args.seq_len, args.seq_len // CHUNK), flush=True)

    target = out_dir / "onnx" / ("model_L%d.onnx" % args.seq_len)
    names = ["input_ids", "attention_mask", "position_ids", "read_positions",
             "image_features", "image_slots"]
    started = time.time()
    # Only read_positions is dynamic: the sequence length must stay static so a
    # wrong-length call fails instead of silently skipping chunks.
    if args.mode == "dynamo":
        torch.onnx.export(
            graph, (ids, mask, positions, reads, feats, slots), str(target),
            input_names=names, output_names=["logits"],
            dynamic_shapes={
                "input_ids": {}, "attention_mask": {}, "position_ids": {},
                "read_positions": {0: "reads"},
                "image_features": {0: "visual"}, "image_slots": {0: "visual"},
            },
            opset_version=18, dynamo=True, external_data=True, optimize=args.optimize,
        )
    else:
        torch.onnx.export(
            graph, (ids, mask, positions, reads, feats, slots), str(target),
            input_names=names, output_names=["logits"],
            dynamic_axes={"read_positions": {0: "reads"}, "logits": {1: "reads"},
                          "image_features": {0: "visual"}, "image_slots": {0: "visual"}},
            opset_version=18, do_constant_folding=False, dynamo=False,
        )
    print("exported %s in %.1f s -> %s" % (args.mode, time.time() - started, target), flush=True)

    vision_target = out_dir / "onnx" / "vision_encoder.onnx"
    if not args.skip_vision:
        patch_dim = (3 * model.config.vision_config.temporal_patch_size
                     * model.config.vision_config.patch_size ** 2)
        pixel_values = torch.zeros(256, patch_dim, dtype=next(graph.parameters()).dtype)
        grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
        from export.vision_export_patch import apply as patch_vision

        vision = VisionGraph(patch_vision(model)).eval()
        with torch.no_grad():
            probe = vision(pixel_values, grid)
        print("vision features:", tuple(probe.shape), flush=True)
        started = time.time()
        torch.onnx.export(
            vision, (pixel_values, grid), str(vision_target),
            input_names=["pixel_values", "image_grid_thw"], output_names=["image_features"],
            dynamic_shapes={"pixel_values": {0: "patches"}, "image_grid_thw": {}},
            opset_version=18, dynamo=True, external_data=True, optimize=False,
        )
        print("exported vision encoder in %.1f s -> %s"
              % (time.time() - started, vision_target), flush=True)

    write_meta(out_dir, pad_id, args.dtype)

    # Reference logits for every golden case, at this padded length.
    payload = {}
    for index, c in enumerate(golden["cases"]):
        if len(c["input_ids"]) > args.seq_len:      # does not fit this bucket
            continue
        i_, m_, p_ = padded_inputs(c["input_ids"], args.seq_len, pad_id)
        r_ = torch.tensor(c["read_positions"], dtype=torch.long)
        empty = torch.zeros(0, hidden, dtype=next(graph.parameters()).dtype)
        no_slots = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            payload["case%d" % index] = graph(
                i_, m_, p_, r_, empty, no_slots
            )[0].numpy().astype(np.float32)
    np.savez(ROOT / "artifacts/torch_logits.npz", **payload)   # same keys at any L, so the last export wins
    print("wrote artifacts/torch_logits.npz (%d of %d cases fit L=%d)"
          % (len(payload), len(golden["cases"]), args.seq_len), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
