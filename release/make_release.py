# -*- coding: utf-8 -*-
"""Stage a GitHub Release from models/: split what is too big, hash everything, write manifest.json.

    python release/make_release.py                           # models/ -> dist/release/
    python release/make_release.py --version v1 --max-part-gib 1.9
    python release/make_release.py --models path/to/models   # stage from another checkout's models/

GitHub allows at most 2 GiB per release asset (and 1000 assets per release), so a file larger than ``--max-part-gib`` is cut into
byte parts (``<name>.part000``, ...) that download_model.py joins again and checks against the whole file's sha256. Asset names are
flat, so ``onnx/onnx/model_L512.onnx`` is published as ``onnx__onnx__model_L512.onnx``; manifest.json keeps the real path.
Nothing is uploaded here. Upload dist/release/* yourself, for example:

    gh release create v1 dist/release/* --title "v1" --notes "ONNX files for Intern-Decision-0.8B-ONNX-CPU"

Every bucket on disk (onnx/model_L<N>.onnx: 384, 512, 1024, 1280) is staged. The graphs read ONE shared weights file
(onnx/weights.onnx.data, weights.onnx.data_1; see export/share_external_data.py), so an extra bucket adds only its graph (~0.25 GB).
Every asset stays under --max-part-gib (1.9 GiB); anything larger is split into parts.

What goes in: everything the runtime and the demos read: models/onnx/** (the ONNX graphs, their external data, graph_meta/config)
and the few models/hf files the demos and the licences need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
HF_FILES = ["tokenizer.json", "preprocessor_config.json", "config.json", "LICENSE", "LICENSE-QWEN"]


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of a multi-GB file on the same drive
    except OSError:
        shutil.copy2(src, dst)


def split(src: pathlib.Path, out_dir: pathlib.Path, stem: str, part_bytes: int) -> list:
    parts = []
    with open(src, "rb") as handle:
        index = 0
        while True:
            name = "%s.part%03d" % (stem, index)
            digest, written = hashlib.sha256(), 0
            with open(out_dir / name, "wb") as out:
                while written < part_bytes:
                    block = handle.read(min(CHUNK, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                (out_dir / name).unlink()
                break
            parts.append({"asset": name, "size": written, "sha256": digest.hexdigest()})
            index += 1
            if written < part_bytes:
                break
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--out", default=str(ROOT / "dist" / "release"))
    parser.add_argument("--name", default="intern-decision-0.8b-onnx")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--max-part-gib", type=float, default=1.9,
                        help="assets above this are split (GitHub's limit is 2 GiB)")
    args = parser.parse_args()

    models, out = pathlib.Path(args.models), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = sorted(p for p in (models / "onnx").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    chosen += [models / "hf" / name for name in HF_FILES if (models / "hf" / name).exists()]
    if not chosen:
        raise SystemExit("nothing under %s (run export/prepare_model.py first)" % models)
    limit = int(args.max_part_gib * (1 << 30))

    entries = []
    for path in chosen:
        rel = path.relative_to(models).as_posix()
        asset = rel.replace("/", "__")
        size = path.stat().st_size
        entry = {"path": rel, "asset": asset, "size": size, "sha256": sha256_of(path), "parts": None}
        if size > limit:
            entry["parts"] = split(path, out, asset, limit)
            print("split  %-44s %9.1f MB -> %d parts" % (rel, size / 1e6, len(entry["parts"])))
        else:
            link_or_copy(path, out / asset)
            print("asset  %-44s %9.1f MB" % (rel, size / 1e6))
        entries.append(entry)

    buckets = sorted(int(m.group(1)) for e in entries for m in [re.fullmatch(r"(?:.*/)?model_L(\d+)\.onnx", e["path"])] if m)
    manifest = {"format": 1, "name": args.name, "version": args.version, "buckets": buckets, "files": entries}
    print("buckets in the manifest: %s" % ", ".join(map(str, buckets)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (e["sha256"], e["path"]) for e in entries), encoding="utf-8")
    too_big = [a.name for a in out.iterdir() if a.stat().st_size >= (2 << 30)]
    if too_big:
        raise SystemExit("asset(s) of 2 GiB or more: %s" % ", ".join(too_big))
    print("wrote %s: %d files, %d assets" % (out, len(entries), len(list(out.iterdir()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
