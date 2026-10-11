# -*- coding: utf-8 -*-
"""Make the Qwen3.5 vision tower exportable, for one image at a time.

Two places in ``modeling_qwen3_5.py`` call ``.tolist()`` on values derived from
``image_grid_thw``, which makes the graph data-dependent and stops
``torch.export``:

* ``Qwen3_5VisionAttention.forward`` splits the sequence per image using
  ``lengths.tolist()`` so that attention stays inside each image
* ``get_image_features`` splits the output per image using ``split_sizes.tolist()``

Both splits exist only to keep several images apart. With **one image per call**
they are no-ops: the whole patch sequence belongs to that image, so plain
attention over it is exactly what the split path computes, and the output needs
no splitting. The runtime therefore encodes images one at a time.

``apply`` swaps in single-image versions; nothing else about the tower changes.
"""

from __future__ import annotations

import torch


def _single_image_attention_forward(self, hidden_states, cu_seqlens,
                                    position_embeddings=None, max_seqlen=None, **kwargs):
    """Qwen3_5VisionAttention.forward without the per-image split."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    # [1, heads, seq, head_dim]
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=None, is_causal=False,
        scale=self.scaling,
    )
    attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


def apply(model):
    """Patch the vision tower in place; returns the module that owns `visual`."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    mod.Qwen3_5VisionAttention.forward = _single_image_attention_forward

    def single_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        return output.pooler_output  # one image: no split needed

    # `visual` lives on the inner Qwen3_5Model, not the ForConditionalGeneration
    # wrapper, so bind the helper where the module actually is.
    inner = model.model if hasattr(model, "model") and hasattr(model.model, "visual") else model
    type(inner).get_image_features_single = single_image_features
    return inner
