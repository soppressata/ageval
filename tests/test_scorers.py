from __future__ import annotations

import pytest

from ageval.core import (
    Task,
    Prediction,
    Score,
    register_agent,
    get_scorer,
    list_scorers,
)


def T(**kw) -> Task:
    return Task(id="t", input=kw.pop("input", ""), **kw)


def P(o: str = "", err: str | None = None) -> Prediction:
    return Prediction(output=o, error=err)


@pytest.fixture(autouse=True)
def registry_cleanup():
    from ageval.core import _agent_registry, _scorer_registry
    import copy

    saved_agents = copy.copy(_agent_registry)
    saved_scorers = copy.copy(_scorer_registry)
    yield
    _agent_registry.clear()
    _agent_registry.update(saved_agents)
    _scorer_registry.clear()
    _scorer_registry.update(saved_scorers)


# ---------- helpers ----------

def builtin_scorer_names() -> list[str]:
    return [n for n in list_scorers() if n != "llm_judge"]


# ============================================================
# A. Agent-error short-circuit
# ============================================================

@pytest.mark.parametrize("name", list_scorers())
def test_agent_error_short_circuit(name: str) -> None:
    scorer = get_scorer(name)
    score = scorer.score(T(expected="anything"), P("", err="boom"))
    assert score == Score(value=0.0, passed=False, detail="agent error")


# ============================================================
# B. Never-raise
# ============================================================

@pytest.mark.parametrize("name", list_scorers())
def test_never_raise(name: str) -> None:
    scorer = get_scorer(name)
    score = scorer.score(T(expected=object()), P("\x00weird"))
    assert 0.0 <= score.value <= 1.0
    assert isinstance(score.passed, bool)


# ============================================================
# C. exact
# ============================================================

class TestExactScorer:
    def test_default_case_insensitive_strip(self) -> None:
        scorer = get_scorer("exact")
        score = scorer.score(T(expected="Yes"), P(" yes "))
        assert score.value == 1.0
        assert score.passed is True

    def test_case_sensitive_fails(self) -> None:
        scorer = get_scorer("exact", case_sensitive=True)
        score = scorer.score(T(expected="Yes"), P(" yes "))
        assert score.value == 0.0
        assert score.passed is False

    def test_strip_false_fails(self) -> None:
        scorer = get_scorer("exact", strip=False)
        score = scorer.score(T(expected="Yes"), P(" yes "))
        assert score.value == 0.0
        assert score.passed is False

    def test_none_expected(self) -> None:
        scorer = get_scorer("exact")
        score = scorer.score(T(expected=None), P("anything"))
        assert score.value == 0.0
        assert score.passed is False
        assert score.detail == "no expected value"

    def test_exact_match_empty(self) -> None:
        scorer = get_scorer("exact")
        score = scorer.score(T(expected=""), P(""))
        assert score.value == 1.0
        assert score.passed is True


# ============================================================
# D. includes
# ============================================================

class TestIncludesScorer:
    def test_str_expected(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected="hello"), P("hello world"))
        assert score.value == 1.0
        assert score.passed is True

    def test_list_expected_fraction(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected=["a", "b", "c"]), P("a and c"))
        assert score.value == pytest.approx(2 / 3)
        assert score.passed is False
        assert "missing" in score.detail
        assert "b" in score.detail

    def test_list_all_present(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected=["cat", "dog"]), P("the cat and dog"))
        assert score.value == 1.0
        assert score.passed is True
        assert score.detail == ""

    def test_case_insensitive_default(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected="HELLO"), P("say hello"))
        assert score.value == 1.0

    def test_empty_list(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected=[]), P("some output"))
        assert score.value == 0.0
        assert score.passed is False
        assert score.detail == "no expected value"

    def test_missing_detail_mentions_substrings(self) -> None:
        scorer = get_scorer("includes")
        score = scorer.score(T(expected=["foo", "bar"]), P("nothing here"))
        assert "foo" in score.detail
        assert "bar" in score.detail


# ============================================================
# E. regex
# ============================================================

class TestRegexScorer:
    def test_pattern_arg(self) -> None:
        scorer = get_scorer("regex", pattern=r"\d+")
        score = scorer.score(T(), P("abc 123 def"))
        assert score.value == 1.0
        assert score.passed is True
        assert score.extra["group"] == "123"

    def test_expected_fallback(self) -> None:
        scorer = get_scorer("regex")
        score = scorer.score(T(expected=r"cat"), P("the cat sat"))
        assert score.value == 1.0
        assert score.passed is True

    def test_no_pattern_no_expected(self) -> None:
        scorer = get_scorer("regex")
        score = scorer.score(T(expected=None), P("abc"))
        assert score.value == 0.0
        assert score.passed is False
        assert "scoring error" in score.detail

    def test_flag_i(self) -> None:
        scorer = get_scorer("regex", flags="i")
        score = scorer.score(T(expected=r"hello"), P("HELLO there"))
        assert score.value == 1.0

    def test_flag_m(self) -> None:
        scorer = get_scorer("regex", flags="m")
        score = scorer.score(T(expected=r"^world"), P("hello\nworld"))
        assert score.value == 1.0

    def test_flag_s(self) -> None:
        scorer = get_scorer("regex", flags="s")
        score = scorer.score(T(expected=r"a.b"), P("a\nb"))
        assert score.value == 1.0

    def test_invalid_pattern(self) -> None:
        scorer = get_scorer("regex", pattern=r"[invalid")
        score = scorer.score(T(), P("anything"))
        assert score.value == 0.0
        assert score.passed is False
        assert "scoring error" in score.detail

    def test_groups_in_extra(self) -> None:
        scorer = get_scorer("regex", pattern=r"(\d+)-(\d+)")
        score = scorer.score(T(), P("code 42-99"))
        assert score.extra["group"] == "42-99"
        assert score.extra["groups"] == ["42", "99"]

    def test_no_match(self) -> None:
        scorer = get_scorer("regex", pattern=r"zzz")
        score = scorer.score(T(), P("abc"))
        assert score.value == 0.0
        assert score.detail == "no match"


# ============================================================
# F. numeric
# ============================================================

class TestNumericScorer:
    def test_last_number_wins(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected=42), P("first 7 then 42"))
        assert score.value == 1.0
        assert score.passed is True
        assert score.extra["parsed"] == 42.0

    def test_commas(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected=-1234.5), P("value -1,234.5 here"))
        assert score.value == 1.0
        assert score.extra["parsed"] == -1234.5

    def test_scientific(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected=2500), P("result 2.5e3"))
        assert score.value == 1.0

    def test_no_digits_fail(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected=5), P("no numbers here"))
        assert score.value == 0.0
        assert score.passed is False
        assert "no number" in score.detail

    def test_tolerance_absolute(self) -> None:
        scorer = get_scorer("numeric", tolerance=0.5, relative=False)
        score = scorer.score(T(expected=10.0), P("9.7"))
        assert score.value == 1.0
        score2 = scorer.score(T(expected=10.0), P("9.2"))
        assert score2.value == 0.0

    def test_tolerance_relative(self) -> None:
        scorer = get_scorer("numeric", tolerance=0.1, relative=True)
        score = scorer.score(T(expected=100.0), P("105"))
        assert score.value == 1.0
        score2 = scorer.score(T(expected=100.0), P("120"))
        assert score2.value == 0.0

    def test_non_numeric_expected_error(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected="abc"), P("42"))
        assert score.value == 0.0
        assert score.passed is False
        assert "scoring error" in score.detail

    def test_parsed_in_extra(self) -> None:
        scorer = get_scorer("numeric")
        score = scorer.score(T(expected=7), P("value is 7.0"))
        assert score.extra["parsed"] == 7.0


# ============================================================
# G. json_subset
# ============================================================

class TestJsonSubsetScorer:
    def test_plain_json(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"a": 1}), P('{"a": 1, "b": 2}'))
        assert score.value == 1.0
        assert score.passed is True

    def test_fenced(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"x": 10}), P('```json\n{"x": 10, "y": 20}\n```'))
        assert score.value == 1.0

    def test_embedded_balanced(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"a": 1}), P('output is {"a": 1, "b": 2} done'))
        assert score.value == 1.0

    def test_half_match(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"a": 1, "b": 2}), P('{"a": 1, "b": 999}'))
        assert score.value == pytest.approx(0.5)
        assert score.passed is False

    def test_nested_list_extra_ok(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"items": [1, 2]}), P('{"items": [1, 2, 3]}'))
        assert score.value == 1.0
        assert score.passed is True

    def test_unparseable(self) -> None:
        scorer = get_scorer("json_subset")
        score = scorer.score(T(expected={"a": 1}), P("not json at all"))
        assert score.value == 0.0
        assert score.passed is False
        assert "output is not valid JSON" in score.detail
        assert "JSON" in score.detail


# ============================================================
# H. llm_judge
# ============================================================

@register_agent("_fake_judge_t4")
class FakeJudgeT4:
    """A fake judge agent whose output is set per-instance."""

    name = "_fake_judge_t4"

    def __init__(self, output: str = '{"score":0.9,"reason":"good"}') -> None:
        self._output = output

    def predict(self, task: Task) -> Prediction:
        return Prediction(output=self._output)


class TestLlmJudgeScorer:
    def test_good_verdict_passed(self) -> None:
        FakeJudgeT4._class_output = '{"score":0.9,"reason":"good"}'

        def make(**kw):
            return FakeJudgeT4(output='{"score":0.9,"reason":"good"}')

        import ageval.core as core

        saved = core._agent_registry.get("_fake_judge_t4")
        core._agent_registry["_fake_judge_t4"] = make
        try:
            scorer = get_scorer("llm_judge", agent="_fake_judge_t4")
            score = scorer.score(T(input="q", expected="a"), P("answer"))
            assert score.value == 0.9
            assert score.passed is True
            assert score.extra["verdict"] == '{"score":0.9,"reason":"good"}'
        finally:
            if saved is not None:
                core._agent_registry["_fake_judge_t4"] = saved
            else:
                del core._agent_registry["_fake_judge_t4"]

    def test_low_verdict_not_passed(self) -> None:
        def make(**kw):
            return FakeJudgeT4(output='{"score":0.2,"reason":"bad"}')

        import ageval.core as core

        saved = core._agent_registry.get("_fake_judge_t4")
        core._agent_registry["_fake_judge_t4"] = make
        try:
            scorer = get_scorer("llm_judge", agent="_fake_judge_t4")
            score = scorer.score(T(input="q", expected="a"), P("answer"))
            assert score.value == 0.2
            assert score.passed is False
        finally:
            if saved is not None:
                core._agent_registry["_fake_judge_t4"] = saved
            else:
                del core._agent_registry["_fake_judge_t4"]

    def test_unparseable_verdict(self) -> None:
        def make(**kw):
            return FakeJudgeT4(output="i refuse")

        import ageval.core as core

        saved = core._agent_registry.get("_fake_judge_t4")
        core._agent_registry["_fake_judge_t4"] = make
        try:
            scorer = get_scorer("llm_judge", agent="_fake_judge_t4")
            score = scorer.score(T(input="q", expected="a"), P("answer"))
            assert score.value == 0.0
            assert score.passed is False
            assert score.detail == "could not parse judge verdict"
        finally:
            if saved is not None:
                core._agent_registry["_fake_judge_t4"] = saved
            else:
                del core._agent_registry["_fake_judge_t4"]

    def test_judge_error_fails(self) -> None:
        class ErrJudge:
            name = "_fake_judge_t4"

            def __init__(self, **kw):
                pass

            def predict(self, task: Task) -> Prediction:
                return Prediction(error="boom")

        import ageval.core as core

        saved = core._agent_registry.get("_fake_judge_t4")
        core._agent_registry["_fake_judge_t4"] = ErrJudge
        try:
            scorer = get_scorer("llm_judge", agent="_fake_judge_t4")
            score = scorer.score(T(input="q", expected="a"), P("answer"))
            assert score.value == 0.0
            assert score.passed is False
            assert "agent error" in score.detail
        finally:
            if saved is not None:
                core._agent_registry["_fake_judge_t4"] = saved
            else:
                del core._agent_registry["_fake_judge_t4"]

    def test_fenced_verdict(self) -> None:
        def make(**kw):
            return FakeJudgeT4(output='```json\n{"score":0.85,"reason":"ok"}\n```')

        import ageval.core as core

        saved = core._agent_registry.get("_fake_judge_t4")
        core._agent_registry["_fake_judge_t4"] = make
        try:
            scorer = get_scorer("llm_judge", agent="_fake_judge_t4")
            score = scorer.score(T(input="q", expected="a"), P("answer"))
            assert score.value == 0.85
            assert score.passed is True
        finally:
            if saved is not None:
                core._agent_registry["_fake_judge_t4"] = saved
            else:
                del core._agent_registry["_fake_judge_t4"]
