# -*- coding: utf-8 -*-
"""Golden outputs for image requests, from the checkpoint's own inference.py.

Also records the processor's pixel_values and image_grid_thw so the numpy
preprocessing can be checked against them directly, and the prompt string so the
``Picture N:`` / vision-pad expansion can be checked without running the model.

Run in .venv-ref.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models/hf"
IMAGES = ROOT / "artifacts/images"
sys.path.insert(0, str(HF))


def make_images():
    """Small, unambiguous pictures so a wrong answer means a wrong pipeline."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    made = {}

    red = Image.new("RGB", (224, 224), (215, 30, 30))
    red.save(IMAGES / "red.png")
    made["red"] = "red.png"

    blue = Image.new("RGB", (224, 224), (30, 60, 215))
    blue.save(IMAGES / "blue.png")
    made["blue"] = "blue.png"

    circle = Image.new("RGB", (192, 256), "white")
    draw = ImageDraw.Draw(circle)
    draw.ellipse((40, 60, 152, 196), fill="black")
    circle.save(IMAGES / "circle.png")
    made["circle"] = "circle.png"

    return made


CASES = [
    {
        "id": "colour_red",
        "image": "red.png",
        "state": "Look at the attached image.",
        "questions": {
            "colour": {
                "type": "choice",
                "instructions": "What colour fills the image?",
                "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"},
            },
        },
    },
    {
        "id": "colour_blue",
        "image": "blue.png",
        "state": "Look at the attached image.",
        "questions": {
            "colour": {
                "type": "choice",
                "instructions": "What colour fills the image?",
                "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"},
            },
        },
    },
    {
        "id": "shape_circle",
        "image": "circle.png",
        "state": "Look at the attached image.",
        "questions": {
            "shape": {
                "type": "choice",
                "instructions": "What shape is drawn?",
                "criteria": {"circle": "a circle", "square": "a square", "triangle": "a triangle"},
            },
            "dark": {
                "type": "noul",
                "instructions": "Is the shape darker than the background?",
                "criteria": {},
            },
        },
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/golden_images.json")
    args = ap.parse_args()

    import torch  # noqa: F401
    from inference import DecisionEngine

    made = make_images()
    print("test images:", made)

    engine = DecisionEngine(str(HF), device="cpu", dtype="float32",
                            attn_implementation="eager", media_root=str(IMAGES))

    arrays = {}
    records = []
    for case in CASES:
        row = {"state": case["state"], "questions": case["questions"],
               "images": [case["image"]]}
        compiled, batch, positions = engine.backend.encode(row)
        messages = json.loads(json.dumps(compiled.messages))  # deep copy
        # reproduce the text the backend rendered, with {'type': 'image'} parts
        converted = []
        for part in messages[1]["content"]:
            converted.append({"type": "image"} if part.get("type") == "image_url" else part)
        messages[1]["content"] = converted
        text = engine.backend.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False, add_vision_id=True,
        )
        result = engine.predict(row)

        index = len(records)
        arrays["pixel_values%d" % index] = batch["pixel_values"].numpy().astype(np.float32)
        arrays["image_grid_thw%d" % index] = batch["image_grid_thw"].numpy().astype(np.int64)

        records.append({
            "id": case["id"],
            "image": case["image"],
            "request": row,
            "prompt": text,
            "input_ids": batch["input_ids"][0].tolist(),
            "read_positions": positions.tolist(),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
            "visual_tokens": int((batch["input_ids"][0] == engine.backend.processor.tokenizer
                                  .convert_tokens_to_ids("<|image_pad|>")).sum()),
            "fields": list(compiled.fields),
            "answers": result["answers"],
            "input_tokens": result["usage"]["input_tokens"],
        })
        print("  %-14s %s  %4d tokens (%d visual)  grid=%s -> %s"
              % (case["id"], case["image"], records[-1]["input_tokens"],
                 records[-1]["visual_tokens"], records[-1]["image_grid_thw"],
                 {k: v["decision"] for k, v in result["answers"].items()}))

    out = ROOT / args.out
    out.write_text(json.dumps({
        "model": "Intern-Decision-0.8B",
        "temperature": engine.temperature,
        "cases": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(ROOT / "artifacts/golden_images_pixels.npz", **arrays)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
