# -*- coding: utf-8 -*-
"""The Intern-Decision request protocol, without torch or transformers.

Ported from the checkpoint's own ``inference.py``:

* questions compile to a schema block listing one single-token symbol per option
* the assistant turn carries a JSON skeleton whose values are ``<decision>``
* the logits **one position before** each ``<decision>`` token predict its symbol
* probabilities are softmaxed over that field's symbols, then temperature-scaled

Every constant here is taken from the reference implementation rather than
re-derived, because the checkpoint was trained against these exact strings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "ANSWER_SYMBOLS", "DECISION_TOKEN", "DEFAULT_TEMPERATURE", "SYSTEM_PROMPT",
    "CompiledRequest", "compile_request", "options_of", "scale_probabilities",
    "argmax_label",
]

DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DEFAULT_TEMPERATURE = 2.747760550703
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in the "
    "user message to make the requested decisions. For every field, choose exactly one "
    "answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON "
    "object mapping each field name to its chosen symbol. Use the field names and symbols "
    "exactly as given. Do not include explanations, Markdown, or extra text."
)

MAX_QUESTIONS = 16
MAX_IMAGES = 8

_NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
_NOUL_NO = "The answer is no (negative, or disagree with the claim)."


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def options_of(question: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(value, description) pairs in the order the model was trained to see them."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping):
            raise ValueError("choice criteria must be an object")
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(i), str(v)) for i, v in enumerate(criteria)]
        if isinstance(criteria, Mapping):
            return [(str(k), str(v)) for k, v in criteria.items()]
        raise ValueError("score criteria must be a list or object")
    if kind == "noul":
        descriptions = criteria if isinstance(criteria, Mapping) else {}
        yes = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {"yes", "true", "1"}),
            _NOUL_YES,
        )
        no = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {"no", "false", "0"}),
            _NOUL_NO,
        )
        # 'no' first: it owns symbol A.
        return [("no", no), ("yes", yes)]
    raise ValueError("unsupported question type: %r" % (kind,))


@dataclass(frozen=True)
class CompiledRequest:
    """One request rendered into chat messages plus the symbol bookkeeping."""

    messages: List[Dict[str, Any]]
    fields: Tuple[str, ...]
    symbols: Dict[str, Tuple[str, ...]]
    options: Dict[str, List[Tuple[str, str]]]
    image_count: int


def compile_request(request: Mapping[str, Any]) -> CompiledRequest:
    """inference.py :: compile_row, with the validation of validate_request."""
    if not isinstance(request, Mapping) or "state" not in request:
        raise ValueError("supply an object containing state and questions")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError("supply 1-%d questions" % MAX_QUESTIONS)

    images = list(request.get("images") or [])
    if len(images) > MAX_IMAGES:
        raise ValueError("upload 1-%d images" % MAX_IMAGES)

    fields: List[str] = []
    symbols: Dict[str, Tuple[str, ...]] = {}
    options: Dict[str, List[Tuple[str, str]]] = {}
    schema_lines: List[str] = []

    for field, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError("question %r is not an object" % (field,))
        opts = options_of(question)
        if not opts:
            raise ValueError("question %r has no options" % (field,))
        if len(opts) > len(ANSWER_SYMBOLS):
            raise ValueError(
                "at most %d single-token answer symbols are supported" % len(ANSWER_SYMBOLS)
            )
        name = str(field)
        fields.append(name)
        symbols[name] = tuple(ANSWER_SYMBOLS[: len(opts)])
        options[name] = opts
        schema_lines.append("%s: %s" % (name, question.get("instructions", "")))
        for symbol, (value, description) in zip(symbols[name], opts):
            schema_lines.append("    %s = %s: %s" % (symbol, value, description))

    state = _dumps(request.get("state"))
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        "## State\n" + state + "\n## Decision schema\n" + "\n".join(schema_lines)
    )
    if DECISION_TOKEN in user_text:
        raise ValueError("reserved decision marker appears in input evidence")

    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return CompiledRequest(
        messages=messages,
        fields=tuple(fields),
        symbols=symbols,
        options=options,
        image_count=len(images),
    )


def argmax_label(probabilities: Mapping[str, float]) -> str:
    """inference.py :: argmax -- ties break lexicographically by label."""
    return min(probabilities, key=lambda label: (-probabilities[label], label))


def scale_probabilities(
    probabilities: Mapping[str, float], temperature: float
) -> Dict[str, float]:
    """inference.py :: scale_probabilities (temperature applied in log space)."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not probabilities or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()
    ):
        raise ValueError("invalid probability distribution")
    if abs(sum(probabilities.values()) - 1.0) > 0.001:
        raise ValueError("probabilities must sum to one")
    if temperature == 1.0:
        return dict(probabilities)

    logs = {k: (math.log(p) if p else -math.inf) for k, p in probabilities.items()}
    maximum = max(logs.values())
    weights = {k: math.exp((v - maximum) / temperature) for k, v in logs.items()}
    total = sum(weights.values())
    scaled = {k: v / total for k, v in weights.items()}
    if argmax_label(scaled) != argmax_label(probabilities):
        raise ArithmeticError("floating-point temperature scaling changed argmax")
    return scaled


def build_answer(
    kind: str, values: Sequence[str], options: Sequence[Tuple[str, str]],
    probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    """The per-field answer object inference.py returns, after scaling."""
    best = argmax_label(probabilities)
    answer: Dict[str, Any] = {
        "type": kind,
        "probabilities": dict(probabilities),
        "confidence": probabilities[best],
    }
    if kind == "noul":
        answer["noul"] = probabilities["yes"]
    elif kind == "score":
        answer["score"] = sum(float(v) * probabilities[v] for v in values)
        answer["legend"] = dict(options)
    else:
        answer["choice"] = best
    answer["source"] = "local"
    answer["decision"] = best
    return answer
