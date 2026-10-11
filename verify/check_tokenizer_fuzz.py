# -*- coding: utf-8 -*-
"""Oracle: pure-Python Qwen BPE vs the Rust `tokenizers` library.

Run in .venv-ref, which has `tokenizers` installed. The corpus covers the shapes
the decision prompts actually take, plus the Unicode classes where Python's `re`
and the Rust regex crate are known to disagree.
"""
import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from intern_decide.tokenizer import QwenBPE  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKJSON = ROOT / "models/hf/tokenizer.json"

SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<decision>", "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>",
    "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|repo_name|>",
]

# The 62 answer symbols must each stay a single token; the schema block and the
# JSON skeleton are the exact shapes this checkpoint is fed.
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

WHITESPACE = [
    " ", " ", " ", "　", "\x0b", "\x0c", "\x1c", "\x1d",
    "\x1e", "\x1f", "​", " ", " ", "﻿", "", " ",
]

BASE = [
    "", " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n\n", " \n ", "a\n\nb",
    "a", "A", "Z", "AA", "ZZ", "The", " the", "the ",
    "Question: Choose the best matching option.",
    "A: the customer is angry\nB: the customer is calm",
    "<|im_start|>user\nWhat is in this image?<|im_end|>\n",
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "don't", "it's", "they're", "we've", "I'm", "we'll", "he'd",
    "DON'T", "It'S", "THEY'RE",
    "CamelCase", "ALLCAPS", "snake_case_name", "kebab-case-name", "mixed123digits",
    "0", "42", "3.14", "1,000,000", "2026-09-28", "v2.0.0", "0x1F",
    "user@example.com", "https://example.com/a/b?c=d&e=f", "www.example.co.jp",
    "@handle", "#hashtag", "50%", "$1,234.56", "¥5,400", "€99", "100°C",
    "naïve café résumé", "Straße Zürich", "İstanbul", "ﬁrefly", "ǅungla",
    "日本語のテキストです", "これはテストです。", "漢字かなカナ", "全角　スペース",
    "한국어 텍스트", "Русский текст", "العربية", "עברית", "ไทย", "हिन्दी",
    "emoji \U0001f389 test", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 family", "\U0001f1ef\U0001f1f5 flag", "a\U0001f389b",
    "égalité", "á combining", "①②③", "ⅷ roman", "㍿ square",
    "{\"state\":{\"cart\":3},\"total\":42}", "[1,2,3]", "```python\nx = 1\n```",
    "a" * 40, "ab" * 30, "x" * 100, " " * 20, "\n" * 10,
    "The export button crashes in Safari but works in Chrome. Not blocking.",
]
BASE += SPECIALS
BASE += WHITESPACE
BASE += ["a%sb" % w for w in WHITESPACE]
BASE += ["a%s%sb" % (w, w) for w in WHITESPACE[:8]]
BASE += ["x %s y" % s for s in SPECIALS]
# schema / skeleton shapes this checkpoint actually sees
BASE += ['    %s = value: description' % c for c in ANSWER_SYMBOLS[:20]]
BASE += list(ANSWER_SYMBOLS)
BASE += [
    '{\n    "team": "<decision>",\n    "angry": "<decision>"\n}',
    'Return one answer for every field using the supplied answer symbols.',
    '## State\n"hello"\n## Decision schema\nteam: Which team?\n    A = billing: charges',
    '<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n',
    '<decision>',
    'x<decision>y',
]


ALPHABET = list(
    "abcdeABCDE 0123(),.:;-_|?!@#$%/" + chr(92) + chr(39) + chr(34) + "`~^*+=<>[]{}"
    "éüñçİı"
    "日本語漢字ひらがなカタ"
    "한글Русالعไท"
    "\U0001f389\U0001f44d\U0001f1ef\U0001f1f5‍️"
    " 　  ​﻿\x0b\x0c\x1c\x85"
    "\t\n\r "
    "①ⅷ㍿ﬁǅ́"
)


def random_cases(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(k)))
    for _ in range(n // 4):
        w = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        out.append("<|im_start|>user\n%s %s %s<|im_end|>\n" % (w, rng.choice(SPECIALS), w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=4000)
    args = ap.parse_args()

    rng = random.Random(20260928)
    mine = QwenBPE(str(TOKJSON))
    ref = Tokenizer.from_file(str(TOKJSON))
    cases = list(BASE) + random_cases(args.n, rng)

    bad = []
    for text in cases:
        got = list(mine.encode(text))
        exp = ref.encode(text, add_special_tokens=False).ids
        if got != exp:
            bad.append((text, got, exp))

    print("cases: %d   mismatches: %d" % (len(cases), len(bad)))
    for text, got, exp in bad[:12]:
        print("\n  input : %r" % text)
        print("  got   : %s" % got)
        print("  expect: %s" % exp)
        print("  gotstr: %s" % [ref.id_to_token(i) for i in got])
        print("  expstr: %s" % [ref.id_to_token(i) for i in exp])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
