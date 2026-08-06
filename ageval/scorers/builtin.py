from __future__ import annotations

import json
import re
from typing import Any

from ageval.core import Task, Prediction, Score, clamp01, register_scorer


class BaseScorer:
    """Abstract base scorer implementing the ``score`` template method.

    Subclasses implement ``_score``. This base guarantees two invariants in
    exactly one place:

    1. If ``prediction.error`` is not ``None``, return ``Score(0.  False,
       "agent error")``.
    2. ``score`` never raises; internal failures become
       ``Score(0.0, False, "scoring error: ...")``.
    """

    name: str = ""

    def score(self, task: Task, prediction: Prediction) -> Score:
        try:
            if prediction.error is not None:
                return Score(value=0.0, passed=False, detail="agent error")
            return self._score(task, prediction)
        except Exception as e:
            return Score(value=0.0, passed=False, detail=f"scoring error: {e}")

    def _score(self, task: Task, prediction: Prediction) -> Score:
        raise NotImplementedError


@register_scorer("exact")
class ExactScorer(BaseScorer):
    """Exact match between ``prediction.output`` and ``str(task.expected)``."""

    def __init__(self, strip: bool = True, case_sensitive: bool = False) -> None:
        self.strip = strip
        self.case_sensitive = case_sensitive

    def _score(self, task: Task, prediction: Prediction) -> Score:
        if task.expected is None:
            return Score(value=0.0, passed=False, detail="no expected value")
        expected = str(task.expected)
        output = prediction.output
        if self.strip:
            expected = expected.strip()
            output = output.strip()
        if not self.case_sensitive:
            expected = expected.lower()
            output = output.lower()
        value = 1.0 if output == expected else 0.0
        return Score(value=value, passed=value == 1.0, detail="")


@register_scorer("includes")
class IncludesScorer(BaseScorer):
    """Fraction of expected substrings present in the output."""

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def _score(self, task: Task, prediction: Prediction) -> Score:
        expected = task.expected
        if isinstance(expected, str):
            expected = [expected]
        elif not isinstance(expected, list):
            expected = [expected]
        if len(expected) == 0:
            return Score(value=0.0, passed=False, detail="no expected value")
        output = prediction.output
        if not self.case_sensitive:
            output = output.lower()
        present = 0
        missing: list[str] = []
        for sub in expected:
            needle = str(sub) if self.case_sensitive else str(sub).lower()
            if needle in output:
                present += 1
            else:
                missing.append(str(sub))
        value = present / len(expected)
        passed = value == 1.0
        detail = "" if not missing else f"missing: {missing}"
        return Score(value=clamp01(value), passed=passed, detail=detail)


@register_scorer("regex")
class RegexScorer(BaseScorer):
    """Regex search, pattern falling back to ``str(task.expected)``."""

    def __init__(self, pattern: str | None = None, flags: str = "") -> None:
        self.pattern = pattern
        self.flags = flags

    def _score(self, task: Task, prediction: Prediction) -> Score:
        pattern = self.pattern
        if pattern is None:
            if task.expected is None:
                return Score(
                    value=0.0, passed=False, detail="scoring error: no pattern"
                )
            pattern = str(task.expected)
        flag_val = 0
        for c in self.flags:
            if c == "i":
                flag_val |= re.IGNORECASE
            elif c == "m":
                flag_val |= re.MULTILINE
            elif c == "s":
                flag_val |= re.DOTALL
        try:
            compiled = re.compile(pattern, flag_val)
        except re.error as e:
            return Score(value=0.0, passed=False, detail=f"scoring error: {e}")
        match = compiled.search(prediction.output)
        if match is None:
            return Score(value=0.0, passed=False, detail="no match")
        extra: dict[str, Any] = {"group": match.group(0)}
        groups = match.groups()
        if groups:
            extra["groups"] = list(groups)
        return Score(value=1.0, passed=True, detail="", extra=extra)


@register_scorer("numeric")
class NumericScorer(BaseScorer):
    """Compare the last number in the output against ``float(expected)``."""

    _NUMBER_RE = re.compile(
        r"[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    )

    def __init__(self, tolerance: float = 1e-6, relative: bool = False) -> None:
        self.tolerance = tolerance
        self.relative = relative

    def _score(self, task: Task, prediction: Prediction) -> Score:
        try:
            expected = float(task.expected)
        except (TypeError, ValueError) as e:
            return Score(value=0.0, passed=False, detail=f"scoring error: {e}")
        matches = self._NUMBER_RE.findall(prediction.output)
        if not matches:
            return Score(
                value=0.0, passed=False, detail="no number found in output"
            )
        last = matches[-1].replace(",", "")
        try:
            actual = float(last)
        except ValueError as e:
            return Score(value=0.0, passed=False, detail=f"scoring error: {e}")
        diff = abs(actual - expected)
        if self.relative:
            threshold = self.tolerance * max(1.0, abs(expected))
        else:
            threshold = self.tolerance
        close = diff <= threshold
        value = 1.0 if close else 0.0
        return Score(value=value, passed=close, detail="", extra={"parsed": actual})


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_balanced_json(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` substring."""
    for i, c in enumerate(text):
        if c in "{[":
            opening = c
            closing = "}" if c == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            return None
    return None


@register_scorer("json_subset")
class JsonSubsetScorer(BaseScorer):
    """Recursive subset test of ``task.expected`` against parsed output JSON."""

    def _score(self, task: Task, prediction: Prediction) -> Score:
        data = self._try_parse_json(prediction.output.strip())
        if data is None:
            return Score(
                value=0.0, passed=False, detail="output is not valid JSON"
            )
        matched, total = self._match(task.expected, data)
        if total == 0:
            value = 1.0 if self._structure_matches(task.expected, data) else 0.0
        else:
            value = matched / total
        passed = value == 1.0
        return Score(value=clamp01(value), passed=passed, detail="")

    def _try_parse_json(self, text: str) -> Any | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fence = _FENCE_RE.match(text)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass
        balanced = _extract_balanced_json(text)
        if balanced:
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass
        return None

    def _count_leaves(self, obj: Any) -> int:
        if isinstance(obj, dict):
            return sum(self._count_leaves(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(self._count_leaves(v) for v in obj)
        return 1

    def _match(self, expected: Any, actual: Any) -> tuple[int, int]:
        """Return ``(matched_leaves, total_leaves)``."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return (0, self._count_leaves(expected))
            total = 0
            matched = 0
            for k, v in expected.items():
                leaves = self._count_leaves(v)
                total += leaves
                if k in actual:
                    m, _ = self._match(v, actual[k])
                    matched += m
            return (matched, total)
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) < len(expected):
                return (0, self._count_leaves(expected))
            total = 0
            matched = 0
            for i, v in enumerate(expected):
                leaves = self._count_leaves(v)
                total += leaves
                m, _ = self._match(v, actual[i])
                matched += m
            return (matched, total)
        return (1, 1) if expected == actual else (0, 1)

    def _structure_matches(self, expected: Any, actual: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(
                k in actual and self._structure_matches(v, actual[k])
                for k, v in expected.items()
            )
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) < len(expected):
                return False
            return all(
                self._structure_matches(v, actual[i])
                for i, v in enumerate(expected)
            )
        return expected == actual
