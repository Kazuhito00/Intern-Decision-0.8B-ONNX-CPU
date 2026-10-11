# -*- coding: utf-8 -*-
"""Split a model's external data so no single file exceeds a size limit.

GitHub release assets are capped at 2 GiB per file, and the fp32 decoder's
weights land at about 3.0 GB in one blob. Splitting them across
``model.onnx_data``, ``model.onnx_data_1``, ... keeps every file publishable
without touching a single number: the tensors are the same bytes, only the file
they live in changes. onnx-community ships its larger exports the same way.

    python export/shard_external_data.py models/onnx/onnx/model_L512.onnx
    python export/shard_external_data.py <graph> --limit-mb 1900

Needs requirements-export.txt (the `onnx` package); the runtime does not.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_LIMIT = 1900 * 1024 * 1024  # a little under 2 GiB, for headroom

#: Tensors below this stay inside the graph. Shape inference reads small
#: constants (axes, shapes, padding amounts) while loading, and it cannot
#: follow an external reference -- ONNX Runtime fails with "Cannot parse data
#: from external tensors" if they are moved out. 1 KiB is onnx's own default.
DEFAULT_MIN_SIZE = 1024


def shard(model_path: pathlib.Path, limit: int, min_size: int) -> list[pathlib.Path]:
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    print("loading %s" % model_path.name, flush=True)
    model = onnx.load(str(model_path), load_external_data=True)

    inline = 0
    tensors = []
    for tensor in _get_all_tensors(model):
        if not (tensor.HasField("raw_data") and tensor.raw_data):
            continue
        if len(tensor.raw_data) < min_size:
            # leave it where it is, and make sure it is marked as internal
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
            inline += 1
            continue
        tensors.append(tensor)
    print("  %d small tensors kept inline (< %d bytes)" % (inline, min_size), flush=True)
    total = sum(len(t.raw_data) for t in tensors)
    print("  %d tensors, %.1f GB of weights" % (len(tensors), total / 1024**3), flush=True)

    stem = model_path.name  # e.g. model_L512.onnx -> model_L512.onnx.data
    # Biggest first, so one oversized tensor cannot strand a nearly-full shard.
    order = sorted(range(len(tensors)), key=lambda i: -len(tensors[i].raw_data))

    shards: list[int] = []          # bytes written per shard
    handles: list = []
    written: list[pathlib.Path] = []

    def open_shard() -> int:
        index = len(shards)
        name = "%s.data" % stem if index == 0 else "%s.data_%d" % (stem, index)
        path = model_path.parent / name
        handles.append(open(path, "wb"))
        shards.append(0)
        written.append(path)
        return index

    open_shard()
    try:
        for i in order:
            tensor = tensors[i]
            size = len(tensor.raw_data)
            target = next((s for s in range(len(shards)) if shards[s] + size <= limit), None)
            if target is None:
                target = open_shard()

            offset = shards[target]
            handles[target].write(tensor.raw_data)
            shards[target] += size

            name = written[target].name
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
            del tensor.external_data[:]
            for key, value in (("location", name), ("offset", str(offset)),
                               ("length", str(size))):
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
    finally:
        for handle in handles:
            handle.close()

    onnx.save(model, str(model_path), save_as_external_data=False)
    for path, size in zip(written, shards):
        print("  wrote %-34s %8.1f MB" % (path.name, size / 1024**2))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=pathlib.Path, help="path to the .onnx graph")
    parser.add_argument("--limit-mb", type=int, default=DEFAULT_LIMIT // (1024 * 1024),
                        help="maximum size of one data file (default: 1900)")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE,
                        help="tensors smaller than this stay inside the graph")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit("no such graph: %s" % model_path)

    limit = args.limit_mb * 1024 * 1024
    written = shard(model_path, limit, args.min_size)
    # Any data file left over from a previous, differently-shaped split.
    for stale in sorted(model_path.parent.glob(model_path.name + ".data*")):
        if stale not in written:
            stale.unlink()
            print("  removed stale %s" % stale.name)

    print("\nevery data file is now under %d MB" % args.limit_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
