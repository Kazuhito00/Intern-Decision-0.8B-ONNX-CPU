# -*- coding: utf-8 -*-
"""One measurement per process.

The graph is fixed-length, so cost does not depend on how long the request is --
only on how many questions ride along (they all share one prefill). Runs one
configuration and exits, so ORT's arena cannot drift between configurations.
"""
import argparse
import ctypes
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide import DEFAULT_SEQ_LEN, DecisionEngine, gpu_providers  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def peak_rss_mb():
    """Peak working set in MB; argtypes are required or the handle truncates."""
    try:
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="bucket to load (default: %d)" % DEFAULT_SEQ_LEN)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--gpu", action="store_true", help="CUDA / DirectML if available (falls back to CPU)")
    ap.add_argument("--questions", type=int, default=0,
                    help="0 = use the golden case as-is")
    ap.add_argument("--case", type=int, default=0)
    args = ap.parse_args()

    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    request = golden["cases"][args.case]["request"]
    if args.questions:
        base = next(iter(request["questions"].values()))
        request = {
            "state": request["state"],
            "questions": {"q%02d" % i: base for i in range(args.questions)},
        }

    t0 = time.perf_counter()
    engine = DecisionEngine(
        ROOT / args.model,
        tokenizer_json=ROOT / "models/hf/tokenizer.json",
        seq_len=args.seq_len,
        intra_op_threads=args.threads,
        providers=gpu_providers() if args.gpu else None,
    )
    load_s = time.perf_counter() - t0

    result = engine.predict(request)  # warm
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        engine.predict(request)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(json.dumps({
        "graph_seq_len": engine.seq_len,
        "threads": args.threads,
        "provider": engine.provider,
        "runs": args.runs,
        "questions": len(request["questions"]),
        "prompt_tokens": result["usage"]["input_tokens"],
        "load_s": round(load_s, 2),
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(times[0], 1),
        "max_ms": round(times[-1], 1),
        "peak_rss_mb": peak_rss_mb(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
