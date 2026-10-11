# -*- coding: utf-8 -*-
"""Make every sequence-length bucket read ONE shared weights file.

The decoder's weights do not depend on the traced length L: only the graph (the unrolled
linear-attention chunk loop) does. Exporting L=384, 512, 1024 and 1280 separately would
write ~3 GB of identical bytes four times. ONNX external data is just (location, offset,
length), so the graphs can point at the same bytes. Nothing numeric changes: a tensor is
redirected only when its bytes are identical (sha256 and length) to one already in the pool.

    python export/share_external_data.py --adopt models/onnx/onnx/model_L512.onnx
        one-off: rename model_L512.onnx.data* -> weights.onnx.data* and repoint the graph
    python export/share_external_data.py models/onnx/onnx/model_L384.onnx
        a fresh export: tensors already in the pool are repointed, small constants go
        inline (ORT cannot shape-infer through an external reference), anything left over
        (nothing, normally) stays in model_L384.onnx.data

The pool files are called weights.onnx.data, weights.onnx.data_1, ... (neutral names, so a
download of one bucket does not depend on another bucket's files). Needs the `onnx` package.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

POOL = "weights.onnx.data"
MIN_SIZE = 1024                      # smaller tensors stay inside the graph (see shard_external_data.py)
LIMIT = 1900 * 1024 * 1024           # one data file stays under GitHub's 2 GiB asset limit


def _external(tensor) -> dict:
    return {e.key: e.value for e in tensor.external_data}


def _set_external(tensor, onnx, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value


def _read(directory: pathlib.Path, info: dict) -> bytes:
    with open(directory / info["location"], "rb") as handle:
        handle.seek(int(info.get("offset", 0)))
        return handle.read(int(info["length"]))


def _tensors(model):
    from onnx.external_data_helper import _get_all_tensors

    return list(_get_all_tensors(model))


def adopt(graph: pathlib.Path) -> None:
    import onnx

    model = onnx.load(str(graph), load_external_data=False)
    renames = {}
    for tensor in _tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = _external(tensor)
        old = info["location"]
        if old not in renames:
            suffix = old[len(graph.name + ".data"):]          # "" or "_1" ...
            renames[old] = POOL + suffix
    for old, new in renames.items():
        if (graph.parent / new).exists():
            raise SystemExit("%s already exists" % new)
    for tensor in _tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = _external(tensor)
            _set_external(tensor, onnx, renames[info["location"]],
                          int(info.get("offset", 0)), int(info["length"]))
    onnx.save(model, str(graph), save_as_external_data=False)
    for old, new in renames.items():
        (graph.parent / old).rename(graph.parent / new)
        print("  %s -> %s" % (old, new))
    print("%s now reads the shared pool (%d files)" % (graph.name, len(renames)))


def build_pool_index(directory: pathlib.Path, skip: pathlib.Path) -> dict:
    """(sha256, length) -> (location, offset) for every tensor the other graphs keep in the pool."""
    import onnx

    index = {}
    for other in sorted(directory.glob("model_L*.onnx")):
        if other == skip:
            continue
        model = onnx.load(str(other), load_external_data=False)
        for tensor in _tensors(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = _external(tensor)
            if not info["location"].startswith(POOL):
                continue
            key = (hashlib.sha256(_read(directory, info)).hexdigest(), int(info["length"]))
            index.setdefault(key, (info["location"], int(info.get("offset", 0))))
        if index:
            break                      # one graph already in the pool is the whole pool
    return index


def share(graph: pathlib.Path) -> None:
    import onnx

    directory = graph.parent
    index = build_pool_index(directory, graph)
    if not index:
        raise SystemExit("no pooled graph in %s: run --adopt on an existing bucket first" % directory)
    print("pool: %d tensors" % len(index), flush=True)

    model = onnx.load(str(graph), load_external_data=False)
    own = graph.name + ".data"
    inline = shared = kept = 0
    keep_bytes = 0
    stash = []                                   # (tensor, bytes) not found in the pool
    for tensor in _tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = _external(tensor)
        data = _read(directory, info)
        if len(data) < MIN_SIZE:
            tensor.raw_data = data
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
            inline += 1
            continue
        hit = index.get((hashlib.sha256(data).hexdigest(), len(data)))
        if hit:
            _set_external(tensor, onnx, hit[0], hit[1], len(data))
            shared += 1
        else:
            stash.append((tensor, data))
            keep_bytes += len(data)
            kept += 1
    print("  %d shared with the pool, %d small kept inline, %d unique (%.1f MB)"
          % (shared, inline, kept, keep_bytes / 1024**2), flush=True)

    old_files = {p for p in directory.glob(graph.name + ".data*")}
    handles, sizes, names = [], [], []
    try:
        for tensor, data in stash:
            if not handles or sizes[-1] + len(data) > LIMIT:
                name = own if not names else "%s_%d" % (own, len(names))
                names.append(name)
                sizes.append(0)
                # unique bytes go to a temp name first: the source file may be the one being read
                handles.append(open(directory / (name + ".new"), "wb"))
            offset = sizes[-1]
            handles[-1].write(data)
            sizes[-1] += len(data)
            _set_external(tensor, onnx, names[-1], offset, len(data))
    finally:
        for handle in handles:
            handle.close()
    onnx.save(model, str(graph), save_as_external_data=False)
    for path in old_files:
        path.unlink()
    for name in names:
        (directory / (name + ".new")).rename(directory / name)
    print("%s: shares %d tensors with the pool%s"
          % (graph.name, shared, "; own data %s" % ", ".join(names) if names else "; no data file of its own"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", type=pathlib.Path)
    parser.add_argument("--adopt", action="store_true",
                        help="turn this graph's (already sharded) data files into the shared pool")
    args = parser.parse_args()
    graph = args.graph.resolve()
    if not graph.exists():
        raise SystemExit("no such graph: %s" % graph)
    adopt(graph) if args.adopt else share(graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
