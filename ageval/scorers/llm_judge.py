from __future__ import annotations

import json
import re
from typing import Any

from ageval.core import Task, Prediction, Score, clamp01, get_agent, register_scorer

from ageval.scorers.builtin import BaseScorer, _extract_balanced_json

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@register_scorer("llm_judge")
class LlmJudgeScorer(BaseScorer):
    """Score by asking another agent to judge the prediction."""

    def __init__(
        self,
        agent: str = "openrouter",
        agent_args: dict[str, Any] | None = None,
        rubric: str | None = None,
        threshold: float = 0.7,
    ) -> None:
        self.agent_name = agent
        self.agent_args: dict[str, Any] = agent_args if agent_args is not None else {}
        self.rubric = rubric
        self.threshold = threshold
        self._judge: Any = None

    def _score(self, task: Task, prediction: Prediction) -> Score:
        if self._judge is None:
            self._judge = get_agent(self.agent_name, **self.agent_args)
        rubric = (
            self.rubric
            if self.rubric is not None
            else (
                "Judge the answer on a scale from 0.0 (completely wrong) to "
                "1.0 (fully correct), comparing it to the expected answer."
            )
        )
        prompt = (
            f"{rubric}\n\n"
            f"Question: {task.input}\n"
            f"Expected answer: {task.expected}\n"
            f"Actual answer: {prediction.output}\n\n"
            "Reply with ONLY a JSON object in the form "
            '{"score": <float 0..1>, "reason": "<short>"}.'
        )
        synth = Task(id="llm_judge", input=prompt)
        verdict = self._judge.predict(synth)
        if verdict.error is not None:
            return Score(
                value=0.0, passed=False, detail=f"agent error: {verdict.error}"
            )
        value = self._parse_verdict(verdict.output)
        if value is None:
            return Score(
                value=0.0,
                passed=False,
                detail="could not parse judge verdict",
                extra={"verdict": verdict.output},
            )
        value = clamp01(value)
        passed = value >= self.threshold
        return Score(value=value, passed=passed, detail="", extra={"verdict": verdict.output})

    def _parse_verdict(self, text: str) -> float | None:
        """Tolerantly parse ``{"score": <float>, ...}`` from the verdict text."""
        candidates = [text]
        fence = _FENCE_RE.match(text)
        if fence:
            candidates.append(fence.group(1).strip())
        balanced = _extract_balanced_json(text)
        if balanced and balanced not in candidates:
            candidates.append(balanced)
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "score" in obj:
                try:
                    return float(obj["score"])
                except (TypeError, ValueError):
                    continue
        return None
