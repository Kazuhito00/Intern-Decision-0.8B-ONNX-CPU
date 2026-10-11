# -*- coding: utf-8 -*-
"""Gate 1-2: our prompt string and input_ids vs the checkpoint's own inference.py.

Runs with the minimal dependencies only -- golden.json carries the reference
strings and ids, so no torch or transformers is needed here. Any difference
means the model would be fed something it was not trained on.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide.tokenizer import QwenBPE  # noqa: E402
from intern_decide.chat import render  # noqa: E402
from intern_decide.protocol import compile_request  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main():
    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    tokenizer = QwenBPE(str(ROOT / "models/hf/tokenizer.json"))
    marker_id = golden["decision_token_id"]

    failures = 0
    for case in golden["cases"]:
        compiled = compile_request(case["request"])
        prompt = render(compiled.messages)
        ids = tokenizer.encode(prompt)
        positions = [i - 1 for i, t in enumerate(ids) if t == marker_id]

        ok_text = prompt == case["prompt"]
        ok_ids = ids == case["input_ids"]
        ok_pos = positions == case["read_positions"]
        ok_fields = list(compiled.fields) == case["fields"]
        ok_syms = {k: list(v) for k, v in compiled.symbols.items()} == case["symbols"]
        good = ok_text and ok_ids and ok_pos and ok_fields and ok_syms
        failures += not good

        print(
            "%s %-16s prompt=%s ids=%s positions=%s fields=%s symbols=%s  (%d tokens)"
            % ("ok " if good else "XX ", case["id"],
               "OK" if ok_text else "DIFF", "OK" if ok_ids else "DIFF",
               "OK" if ok_pos else "DIFF", "OK" if ok_fields else "DIFF",
               "OK" if ok_syms else "DIFF", len(ids))
        )
        if not ok_text:
            k = first_diff(prompt, case["prompt"])
            print("    first text diff at %d" % k)
            print("      ours  : %r" % prompt[max(0, k - 60):k + 60])
            print("      golden: %r" % case["prompt"][max(0, k - 60):k + 60])
        elif not ok_ids:
            k = first_diff(ids, case["input_ids"])
            print("    len ours=%d golden=%d, first id diff at %d" % (len(ids), len(case["input_ids"]), k))
            print("      ours  : %s" % ids[max(0, k - 4):k + 6])
            print("      golden: %s" % case["input_ids"][max(0, k - 4):k + 6])
        elif not ok_pos:
            print("    positions ours=%s golden=%s" % (positions, case["read_positions"]))

    print("\nfailures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
