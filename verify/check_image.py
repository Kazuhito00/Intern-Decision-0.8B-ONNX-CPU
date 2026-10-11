# -*- coding: utf-8 -*-
"""Gate: our numpy+Pillow preprocessing vs the processor's, and the image prompt.

Compares against the pixel_values / image_grid_thw the reference implementation
actually produced (saved by make_image_golden.py), so no transformers is needed
here -- only Pillow on top of the runtime dependencies.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide.tokenizer import QwenBPE  # noqa: E402
from intern_decide.chat import IMAGE_PLACEHOLDER, render  # noqa: E402
from intern_decide.image import ImageConfig, prepare_images  # noqa: E402
from intern_decide.protocol import compile_request  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "artifacts/images"


def main():
    golden = json.loads((ROOT / "artifacts/golden_images.json").read_text(encoding="utf-8"))
    pixels = np.load(ROOT / "artifacts/golden_images_pixels.npz")
    config = ImageConfig.from_pretrained(ROOT / "models/hf")
    tokenizer = QwenBPE(str(ROOT / "models/hf/tokenizer.json"))
    print("image config:", config)

    failures = 0
    worst_pixel = 0.0
    for index, case in enumerate(golden["cases"]):
        prepared = prepare_images([str(IMAGES / case["image"])], config)
        ref_pixels = pixels["pixel_values%d" % index]
        ref_grid = pixels["image_grid_thw%d" % index]

        ok_grid = prepared.image_grid_thw.tolist() == ref_grid.tolist()
        ok_shape = prepared.pixel_values.shape == ref_pixels.shape
        delta = -1.0
        if ok_shape:
            delta = float(np.abs(prepared.pixel_values - ref_pixels).max())
            worst_pixel = max(worst_pixel, delta)

        ok_tokens = prepared.token_counts[0] == case["visual_tokens"]

        # prompt: the image part comes first, then the text, per _message_content
        compiled = compile_request(case["request"])
        messages = [dict(m) for m in compiled.messages]
        messages[1] = {
            "role": "user",
            "content": IMAGE_PLACEHOLDER + messages[1]["content"],
        }
        prompt = render(messages, prepared.token_counts)
        ids = tokenizer.encode(prompt)
        # The stored prompt is what apply_chat_template emits: a single
        # <|image_pad|>, which the processor expands during tokenization. We
        # expand up front instead, so compare against the expanded form.
        expanded = case["prompt"].replace(
            "<|image_pad|>", "<|image_pad|>" * prepared.token_counts[0], 1
        )
        ok_prompt = prompt == expanded
        ok_ids = ids == case["input_ids"]

        good = ok_grid and ok_shape and ok_tokens and ok_prompt and ok_ids and delta < 0.02
        failures += not good
        print("%s %-14s grid=%s shape=%s tokens=%s prompt=%s ids=%s  max|dp|=%.4f"
              % ("ok " if good else "XX ", case["id"],
                 "OK" if ok_grid else "DIFF", "OK" if ok_shape else "DIFF",
                 "OK" if ok_tokens else "DIFF", "OK" if ok_prompt else "DIFF",
                 "OK" if ok_ids else "DIFF", delta))
        if not ok_prompt:
            for k, (a, b) in enumerate(zip(prompt, case["prompt"])):
                if a != b:
                    print("      first text diff at %d" % k)
                    print("        ours  : %r" % prompt[max(0, k - 70):k + 70])
                    print("        golden: %r" % case["prompt"][max(0, k - 70):k + 70])
                    break
            else:
                print("      lengths %d vs %d" % (len(prompt), len(case["prompt"])))

    print("\nworst |pixel delta|: %.4f (range is [-1, 1])" % worst_pixel)
    print("failures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
