# -*- coding: utf-8 -*-
"""Intern-Decision on onnxruntime + numpy.

One prefill answers every question in the request. The exported graph returns
only the logits at the positions the protocol reads -- one position before each
``<decision>`` marker -- and those are softmaxed over the field's answer symbols
and temperature-scaled, exactly as the checkpoint's inference.py does.

Runtime dependencies: onnxruntime and numpy. Pillow only if images are used.
"""

from __future__ import annotations

import gc
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from .tokenizer import QwenBPE
from .chat import IMAGE_PLACEHOLDER, render
from .chat import position_ids as build_position_ids
from .protocol import (
    ANSWER_SYMBOLS,
    DECISION_TOKEN,
    DEFAULT_TEMPERATURE,
    build_answer,
    compile_request,
    scale_probabilities,
)

__all__ = ["DecisionEngine", "MODEL_NAME", "DEFAULT_SEQ_LEN", "gpu_providers"]

MODEL_NAME = "Intern-Decision-0.8B"

#: The bucket loaded unless asked otherwise. A one-question prompt with a 224 px image is
#: ~260-375 tokens; 384 holds all of them and costs less per call than 512 (cost ~ L).
DEFAULT_SEQ_LEN = 384

_BUCKET = re.compile(r"model_L(\d+)\.onnx")


def find_buckets(onnx_dir: pathlib.Path) -> Dict[int, pathlib.Path]:
    """Sequence-length buckets present on disk: ``model_L<N>.onnx`` -> {N: path}."""
    found = {}
    for path in pathlib.Path(onnx_dir).glob("model_L*.onnx"):
        match = _BUCKET.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]


def _session(path: pathlib.Path, threads: Optional[int],
             providers: Optional[Sequence[str]] = None) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError("%s not found (run download_model.py, or export/prepare_model.py)" % path)
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    wanted = list(providers) if providers else ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(path), sess_options=options, providers=wanted)
    except Exception as exc:                        # a GPU that is too small / missing DLLs: say so and run on the CPU
        if wanted == ["CPUExecutionProvider"]:
            raise
        print("warning: %s could not load %s (%s); using the CPU"
              % (wanted[0], path.name, str(exc).splitlines()[0][:160]), file=sys.stderr)
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class DecisionEngine:
    """Load once, then score structured requests in one forward pass each."""

    def __init__(
        self,
        model_dir: Union[str, pathlib.Path],
        *,
        graph: Optional[str] = None,
        seq_len: Optional[int] = DEFAULT_SEQ_LEN,
        tokenizer_json: Optional[Union[str, pathlib.Path]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        intra_op_threads: Optional[int] = None,
        media_root: Optional[Union[str, pathlib.Path]] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        """``providers``: onnxruntime execution providers, e.g. ``gpu_providers()`` (default: CPU only).

        ``seq_len``: the bucket to load (``model_L<seq_len>.onnx``; default 384). If it does not
        exist, the smallest larger bucket on disk is used. A request that does not fit the loaded
        bucket makes the engine swap to the smallest larger bucket that exists and fits (the old
        graph is released first, so only one decoder is in memory); the engine then stays there.
        ``graph``: an explicit graph file name; it wins over ``seq_len`` and is never swapped.
        """
        model_dir = pathlib.Path(model_dir)
        self.model_dir = model_dir
        self.temperature = float(temperature)

        # The graph is traced at one fixed prompt length: the linear-attention
        # chunk scan is a Python loop, so it unrolls at export time and cannot
        # be made dynamic. Requests are right-padded up to that length.
        meta_path = model_dir / "graph_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.pad_token_id: int = int(meta.get("pad_token_id", 248044))
        self._threads = intra_op_threads
        self._providers = providers
        self._pinned = graph is not None          # an explicit graph= is used as it is
        self.session = None

        if graph is not None:
            path = model_dir / "onnx" / graph
        else:
            want = int(seq_len if seq_len is not None else meta.get("seq_len", DEFAULT_SEQ_LEN))
            buckets = find_buckets(model_dir / "onnx")
            larger = [length for length in buckets if length >= want]
            if larger:
                path = buckets[min(larger)]
            elif buckets:
                raise FileNotFoundError(
                    "no bucket of L>=%d in %s (have: %s); export one with export/prepare_model.py --seq-len %d"
                    % (want, model_dir / "onnx", ", ".join(map(str, buckets)), want))
            else:                                  # an old layout: whatever graph_meta.json names
                path = model_dir / "onnx" / meta.get("graph", "model.onnx")
        self._load_graph(path, meta)

        # The vision graph is optional: a text-only build simply omits it.
        self.vision = None
        vision_name = meta.get("vision_graph")
        if vision_name and (model_dir / "onnx" / vision_name).exists():
            self.vision = _session(model_dir / "onnx" / vision_name, intra_op_threads, providers)
        self.tokenizer = QwenBPE(str(tokenizer_json or model_dir / "tokenizer.json"))

        marker = self.tokenizer.encode(DECISION_TOKEN)
        if len(marker) != 1:
            raise ValueError("the tokenizer does not carry a single-token %s" % DECISION_TOKEN)
        self.marker_id = marker[0]

        # Every answer symbol must stay one token; resolve them once.
        self.symbol_ids: Dict[str, int] = {}
        for symbol in ANSWER_SYMBOLS:
            ids = self.tokenizer.encode(symbol)
            if len(ids) != 1:
                raise ValueError("answer symbol %r is not a single token" % symbol)
            self.symbol_ids[symbol] = ids[0]

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.image_token_id: int = config["image_token_id"]
        self.merge_size: int = config["vision_config"]["spatial_merge_size"]
        self.hidden_size: int = config["text_config"]["hidden_size"]
        self.media_root = pathlib.Path(media_root or ".")
        self._image_config = None

    def _load_graph(self, path: pathlib.Path, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Replace the decoder session. The old one is released first: two 3 GB graphs never coexist."""
        self.session = None
        gc.collect()
        self.session = _session(path, self._threads, self._providers)
        self.graph_name = path.name
        # The graph's own input shape is the truth about L (graph_meta.json may describe another bucket).
        length = self.session.get_inputs()[0].shape[1]
        if not isinstance(length, int):
            match = _BUCKET.fullmatch(path.name)
            length = int(match.group(1)) if match else int((meta or {}).get("seq_len", 512))
        self.seq_len = int(length)
        self.max_length = self.seq_len

    @property
    def buckets(self) -> List[int]:
        """Sequence lengths available on disk."""
        return list(find_buckets(self.model_dir / "onnx"))

    def _fit(self, tokens: int) -> None:
        """Make sure the loaded graph holds ``tokens``; else load the smallest larger bucket that fits."""
        if tokens <= self.seq_len:
            return
        needed = -(-tokens // 64) * 64
        bigger = [] if self._pinned else [n for n in self.buckets if n >= tokens]
        if not bigger:
            raise ValueError(
                "request is %d tokens but this graph is fixed at %d; truncation is "
                "forbidden -- re-export with --seq-len %d"
                % (tokens, self.seq_len, needed)
            )
        self._load_graph(find_buckets(self.model_dir / "onnx")[min(bigger)])

    @property
    def provider(self) -> str:
        """The execution provider the main graph actually runs on."""
        return self.session.get_providers()[0]

    @property
    def image_config(self):
        """Lazy: importing image.py pulls in Pillow, which text-only use avoids."""
        if self._image_config is None:
            from .image import ImageConfig

            self._image_config = ImageConfig.from_pretrained(self.model_dir)
        return self._image_config

    # ------------------------------------------------------------- encoding

    def encode(self, request: Mapping[str, Any]):
        """Compile, render and tokenise one request.

        Returns (compiled, input_ids, read_positions, prepared_images).
        """
        compiled = compile_request(request)

        prepared = None
        messages = compiled.messages
        if compiled.image_count:
            if self.vision is None:
                raise NotImplementedError(
                    "this build has no vision encoder; re-export without --skip-vision"
                )
            from .image import prepare_images  # Pillow only needed for images

            sources = list(request.get("images") or [])
            root = pathlib.Path(request.get("media_root") or self.media_root)
            resolved = [
                s if not isinstance(s, (str, pathlib.Path))        # PIL image / bytes: used as they are
                or (isinstance(s, str) and s.startswith("data:image/"))
                or pathlib.Path(str(s)).is_absolute() else str(root / str(s))
                for s in sources
            ]
            prepared = prepare_images(resolved, self.image_config)
            # inference.py::_message_content puts the images before the text.
            messages = [dict(m) for m in compiled.messages]
            messages[1] = {
                "role": "user",
                "content": IMAGE_PLACEHOLDER * compiled.image_count + messages[1]["content"],
            }

        counts = prepared.token_counts if prepared is not None else []
        prompt = render(messages, counts)
        input_ids = self.tokenizer.encode(prompt)
        self._fit(len(input_ids))
        reads = [i - 1 for i, token in enumerate(input_ids) if token == self.marker_id]
        if len(reads) != len(compiled.fields) or any(p < 0 for p in reads):
            raise ValueError("decision marker count or position mismatch")
        return compiled, input_ids, reads, prepared

    def image_features(self, prepared) -> np.ndarray:
        """Encode each image separately: the graph is traced for one image."""
        rows = []
        offset = 0
        patches_per = (prepared.image_grid_thw.prod(axis=1)).tolist()
        for index, count in enumerate(patches_per):
            chunk = prepared.pixel_values[offset : offset + count]
            offset += count
            grid = prepared.image_grid_thw[index : index + 1]
            rows.append(self.vision.run(["image_features"], {
                "pixel_values": chunk.astype(np.float32),
                "image_grid_thw": grid.astype(np.int64),
            })[0])
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 1), dtype=np.float32)

    # -------------------------------------------------------------- forward

    def logits(self, input_ids: Sequence[int], reads: Sequence[int], prepared=None) -> np.ndarray:
        """Logits at the read positions, shape [len(reads), vocab].

        The prompt is right-padded to the graph's fixed length. Every read
        position precedes the padding and the model is causal, so the padding
        cannot reach them (measured at 1.9e-05 logits; see check_padding.py).
        """
        real = list(input_ids)
        self._fit(len(real))
        pad = self.seq_len - len(real)
        ids = np.asarray([real + [self.pad_token_id] * pad], dtype=np.int64)
        mask = np.asarray([[1] * len(real) + [0] * pad], dtype=np.int64)

        if prepared is not None and len(prepared.image_grid_thw):
            grids = prepared.image_grid_thw
            features = self.image_features(prepared).astype(np.float32)
            slots = np.flatnonzero(np.asarray(real) == self.image_token_id).astype(np.int64)
            if slots.size != features.shape[0]:
                raise RuntimeError(
                    "vision encoder produced %d features for %d image tokens"
                    % (features.shape[0], slots.size)
                )
            # M-RoPE: images occupy a (t, h, w) grid rather than a run of
            # consecutive positions, so the real tokens need get_rope_index.
            positions = build_position_ids(real, grids, self.image_token_id, self.merge_size)
            tail = np.arange(pad, dtype=np.int64) + (int(positions.max()) + 1 if pad else 0)
            positions = np.concatenate(
                [positions, np.broadcast_to(tail, (3, 1, pad))], axis=2
            )
        else:
            features = np.zeros((0, self.hidden_size), dtype=np.float32)
            slots = np.zeros((0,), dtype=np.int64)
            positions = np.arange(self.seq_len, dtype=np.int64)[None, None, :].repeat(3, axis=0)

        feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": np.ascontiguousarray(positions),
            "read_positions": np.asarray(list(reads), dtype=np.int64),
            "image_features": features,
            "image_slots": slots,
        }
        return self.session.run(["logits"], feeds)[0][0].astype(np.float64)

    # --------------------------------------------------------------- decide

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """One request in, one JSON-serialisable response out."""
        compiled, input_ids, reads, prepared = self.encode(request)
        scores = self.logits(input_ids, reads, prepared)

        answers: Dict[str, Any] = {}
        for index, field in enumerate(compiled.fields):
            options = compiled.options[field]
            values = [value for value, _ in options]
            columns = np.asarray(
                [self.symbol_ids[s] for s in compiled.symbols[field]], dtype=np.int64
            )
            row = scores[index, columns]
            row = np.exp(row - row.max())
            row /= row.sum()
            raw = {value: float(p) for value, p in zip(values, row)}
            scaled = scale_probabilities(raw, self.temperature)
            kind = request["questions"][field]["type"]
            answers[field] = build_answer(kind, values, options, scaled)

        return {
            "answers": answers,
            "usage": {
                "input_tokens": len(input_ids),
                "output_tokens": len(answers),
                "decision_count": len(answers),
            },
            "calibration": {"method": "temperature-scaling", "temperature": self.temperature},
            "model": MODEL_NAME,
            "backend": "onnxruntime",
        }
