# -*- coding: utf-8 -*-
"""Qwen3.5 chat template and M-RoPE positions, without jinja2 or transformers.

``render`` covers exactly the branch Intern-Decision uses: a system turn, one
user turn (optionally with images), and a **final assistant turn that carries
content** -- the JSON skeleton -- with ``add_generation_prompt=False``.

That last point is the subtle one. Because the assistant message sits after the
last user message, the template's ``loop.index0 > ns.last_query_index`` test is
true, so it emits an empty think block before the content::

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{content}<|im_end|>\\n

``position_ids`` ports ``Qwen3_5Model.get_rope_index`` / ``get_vision_position_ids``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["IMAGE_PLACEHOLDER", "render", "position_ids", "token_type_ids"]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"

#: What a caller puts in message text where an image belongs.
IMAGE_PLACEHOLDER = "<__media__>"

ROLES = ("system", "user", "assistant")


def render(
    messages: Sequence[Dict[str, str]],
    image_token_counts: Sequence[int] = (),
    add_vision_id: bool = True,
) -> str:
    """Render the compiled messages into the prompt string.

    Each ``IMAGE_PLACEHOLDER`` is replaced, in order, by
    ``<|vision_start|>`` + N ``<|image_pad|>`` + ``<|vision_end|>`` so the image
    features can be scattered into the embeddings. With ``add_vision_id`` the
    template also prefixes ``Picture N: ``, which is what inference.py asks for.
    """
    counts = list(image_token_counts)
    used = 0
    parts: List[str] = []
    last_user = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1
    )
    if last_user < 0:
        raise ValueError("no user message found")

    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ROLES:
            raise ValueError("unsupported role %r" % (role,))
        if role == "system" and index != 0:
            raise ValueError("a system message must come first")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise TypeError("message content must already be rendered to text")

        content = content.strip()  # the template's |trim
        while IMAGE_PLACEHOLDER in content:
            if used >= len(counts):
                raise ValueError("more image placeholders than images")
            block = VISION_START + IMAGE_PAD * counts[used] + VISION_END
            if add_vision_id:
                block = "Picture %d: " % (used + 1) + block
            content = content.replace(IMAGE_PLACEHOLDER, block, 1)
            used += 1

        if role == "assistant" and index > last_user:
            # A trailing assistant turn gets an empty reasoning block first.
            parts.append(IM_START + role + "\n" + EMPTY_THINK + content + IM_END + "\n")
        else:
            parts.append(IM_START + role + "\n" + content + IM_END + "\n")

    if used != len(counts):
        raise ValueError("%d images supplied but %d placeholders found" % (len(counts), used))
    return "".join(parts)


def token_type_ids(input_ids: Sequence[int], image_token_id: int) -> np.ndarray:
    """0 for text, 1 for image tokens -- the modality map M-RoPE groups by."""
    ids = np.asarray(input_ids, dtype=np.int64)
    return (ids == image_token_id).astype(np.int64)


def _runs(types: np.ndarray) -> Iterable[Tuple[int, int, int]]:
    """Yield (type, start, end) for each maximal run of equal values."""
    if types.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(types)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [types.size]))
    for start, end in zip(starts, ends):
        yield int(types[start]), int(start), int(end)


def _vision_positions(start: int, grid: Sequence[int], merge: int) -> np.ndarray:
    """transformers ... Qwen3_5Model.get_vision_position_ids (temp_merge=1)."""
    grid_t, grid_h, grid_w = int(grid[0]), int(grid[1]) // merge, int(grid[2]) // merge
    t = np.arange(grid_t, dtype=np.int64)  # time_interval == 1
    h = np.arange(grid_h, dtype=np.int64) + start
    w = np.arange(grid_w, dtype=np.int64) + start
    T, H, W = np.meshgrid(t, h, w, indexing="ij")
    out = np.stack([T, H, W], axis=0).reshape(3, -1)
    out[0] += start  # after the time_interval multiply, as upstream does
    return out


def position_ids(
    input_ids: Sequence[int],
    image_grid_thw: "np.ndarray | None",
    image_token_id: int,
    merge_size: int = 2,
) -> np.ndarray:
    """3D M-RoPE positions, shape [3, 1, sequence_length]."""
    types = token_type_ids(input_ids, image_token_id)
    grids = [] if image_grid_thw is None or len(image_grid_thw) == 0 else list(image_grid_thw)
    grid_iter = iter(grids)

    chunks: List[np.ndarray] = []
    cursor = 0
    for kind, start, end in _runs(types):
        if kind == 0:
            length = end - start
            chunks.append(
                np.broadcast_to(np.arange(length, dtype=np.int64), (3, length)) + cursor
            )
            cursor += length
        else:
            try:
                grid = next(grid_iter)
            except StopIteration:
                raise ValueError("more image token runs than image grids") from None
            chunks.append(_vision_positions(cursor, grid, merge_size))
            cursor += max(int(grid[1]), int(grid[2])) // merge_size

    if sum(1 for _ in grid_iter):
        raise ValueError("some image grids were never placed in the prompt")
    if not chunks:
        return np.zeros((3, 1, 0), dtype=np.int64)

    positions = np.concatenate(chunks, axis=1)
    if positions.shape[1] != len(input_ids):
        raise ValueError(
            "position_ids length %d does not match %d input ids"
            % (positions.shape[1], len(input_ids))
        )
    return positions[:, None, :].astype(np.int64)
