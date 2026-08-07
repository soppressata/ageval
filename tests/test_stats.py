from __future__ import annotations

import math

import pytest

from ageval.core import Prediction, RunReport, Score, Task, TaskResult
from ageval.stats import (
    RunComparison,
    TaskDelta,
    compare_runs,
    confidence_interval,
    detect_regression,
    effect_size,
    mean_score_delta,
    paired_deltas,
    paired_significance_test,
    pass_rate_delta,
    variance_delta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    task_id: str,
    value: float = 0.0,
    passed: bool = False,
    error: str | None = None,
) -> TaskResult:
    """Build a TaskResult; errored tasks get score=None (per SPEC)."""
    task = Task(id=task_id, input=f"input-{task_id}")
    pred = Prediction(output="output", error=error)
    score = None if error is not None else Score(value=value, passed=passed)
    return TaskResult(task=task, prediction=pred, score=score)


def _report(
    run_id: str,
    specs: list[tuple[str, float, bool, str | None]],
) -> RunReport:
    """Build a RunReport from (tid, value, passed, error) tuples."""
    results = [_result(tid, v, p, e) for (tid, v, p, e) in specs]
    return RunReport(
        run_id=run_id,
        suite_name="suite",
        agent_name="agent",
        started_at="2024-01-01T00:00:00Z",
        results=results,
    )


def _regress_pair() -> tuple[RunReport, RunReport]:
    """Baseline strong, candidate weak -> significant regression."""
    base = _report(
        "base-regress",
        [
            ("t1", 0.90, True, None),
            ("t2", 0.80, False, None),
            ("t3", 0.95, True, None),
            ("t4", 0.85, False, None),
            ("t5", 0.90, True, None),
        ],
    )
    cand = _report(
        "cand-regress",
        [
            ("t1", 0.20, False, None),
            ("t2", 0.15, False, None),
            ("t3", 0.30, False, None),
            ("t4", 0.20, False, None),
            ("t5", 0.25, False, None),
        ],
    )
    return base, cand


def _improve_pair() -> tuple[RunReport, RunReport]:
    """Baseline weak, candidate strong -> significant improvement."""
    base, cand = _regress_pair()
    return cand, base  # swap roles


# ---------------------------------------------------------------------------
# paired_deltas
# ---------------------------------------------------------------------------


def test_paired_deltas_basic_sorted() -> None:
    base = _report(
        "b",
        [("t2", 0.5, False, None), ("t1", 0.8, True, None)],
    )
    cand = _report(
        "c",
        [("t1", 0.9, True, None), ("t2", 0.4, False, None)],
    )
    deltas = paired_deltas(base, cand)
    assert [d.task_id for d in deltas] == ["t1", "t2"]  # sorted by id
    assert isinstance(deltas[0], TaskDelta)
    d1 = deltas[0]
    assert d1.in_baseline and d1.in_candidate
    assert d1.baseline_score == pytest.approx(0.8)
    assert d1.candidate_score == pytest.approx(0.9)
    assert d1.score_delta == pytest.approx(0.1)
    assert d1.baseline_passed is True
    assert d1.candidate_passed is True
    d2 = deltas[1]
    assert d2.score_delta == pytest.approx(-0.1)


def test_paired_deltas_missing_tasks_each_side() -> None:
    base = _report("b", [("a", 0.8, True, None), ("only_b", 0.6, True, None)])
    cand = _report(
        "c", [("only_b", 0.6, True, None), ("only_c", 0.9, True, None)]
    )
    deltas = paired_deltas(base, cand)
    by_id = {d.task_id: d for d in deltas}
    assert set(by_id) == {"a", "only_b", "only_c"}

    a = by_id["a"]
    assert a.in_baseline and not a.in_candidate
    assert a.candidate_score == pytest.approx(0.0)
    assert a.score_delta == pytest.approx(-0.8)

    oc = by_id["only_c"]
    assert oc.in_candidate and not oc.in_baseline
    assert oc.baseline_score == pytest.approx(0.0)
    assert oc.score_delta == pytest.approx(0.9)

    ob = by_id["only_b"]
    assert ob.in_baseline and ob.in_candidate
    assert ob.score_delta == pytest.approx(0.0)


def test_paired_deltas_empty_reports() -> None:
    empty = RunReport(
        run_id="e1", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    empty2 = RunReport(
        run_id="e2", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    assert paired_deltas(empty, empty2) == []
    assert paired_deltas(empty, _report("c", [("t1", 0.5, True, None)]))
    deltas = paired_deltas(empty, _report("c", [("t1", 0.5, True, None)]))
    assert len(deltas) == 1
    assert deltas[0].in_baseline is False
    assert deltas[0].baseline_score == pytest.approx(0.0)
    assert deltas[0].candidate_score == pytest.approx(0.5)


def test_paired_deltas_missing_score_treated_as_zero() -> None:
    # Errored task -> score is None in the report; delta treats it as 0.0.
    base = _report("b", [("t1", 0.8, True, None)])
    cand = _report("c", [("t1", 0.0, False, "boom")])
    deltas = paired_deltas(base, cand)
    d = deltas[0]
    assert d.candidate_score == pytest.approx(0.0)
    assert d.baseline_score == pytest.approx(0.8)
    assert d.score_delta == pytest.approx(-0.8)
    assert d.candidate_passed is False
    assert d.baseline_passed is True


def test_paired_deltas_sorted_deterministically() -> None:
    base = _report(
        "b", [("zeta", 0.1, False, None), ("alpha", 0.2, False, None)]
    )
    cand = _report(
        "c", [("alpha", 0.3, False, None), ("zeta", 0.4, False, None)]
    )
    ids = [d.task_id for d in paired_deltas(base, cand)]
    assert ids == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# compare_runs: structural / edge cases
# ---------------------------------------------------------------------------


def test_compare_runs_signature_and_fields() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert isinstance(comp, RunComparison)
    assert comp.baseline_run_id == "base-regress"
    assert comp.candidate_run_id == "cand-regress"
    assert comp.n_paired == 5
    assert comp.n_baseline_only == 0
    assert comp.n_candidate_only == 0
    expected_fields = {
        "baseline_run_id",
        "candidate_run_id",
        "n_paired",
        "n_baseline_only",
        "n_candidate_only",
        "baseline_pass_rate",
        "candidate_pass_rate",
        "pass_rate_delta",
        "baseline_mean_score",
        "candidate_mean_score",
        "mean_score_delta",
        "mean_delta",
        "variance_delta",
        "std_delta",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "t_statistic",
        "p_value_two_tailed",
        "p_value_greater",
        "cohens_d",
        "regression_detected",
        "tasks",
    }
    for f in expected_fields:
        assert hasattr(comp, f), f"missing field {f}"


def test_compare_runs_empty_reports() -> None:
    empty = RunReport(
        run_id="e1", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    comp = compare_runs(empty, empty)
    assert comp.n_paired == 0
    assert comp.baseline_pass_rate == 0.0
    assert comp.candidate_pass_rate == 0.0
    assert comp.pass_rate_delta == 0.0
    assert comp.mean_delta == 0.0
    assert comp.variance_delta == 0.0
    assert comp.std_delta == 0.0
    assert comp.ci_lower == 0.0
    assert comp.ci_upper == 0.0
    assert comp.t_statistic == 0.0
    assert comp.p_value_two_tailed == 1.0
    assert comp.p_value_greater == 1.0
    assert comp.cohens_d == 0.0
    assert comp.regression_detected is False
    assert comp.tasks == []


def test_compare_runs_no_paired_only_unique_tasks() -> None:
    base = _report("b", [("a", 0.9, True, None), ("shared", 0.5, False, None)])
    cand = _report("c", [("shared", 0.5, False, None), ("c1", 0.8, True, None)])
    comp = compare_runs(base, cand)
    assert comp.n_paired == 1  # only "shared"
    assert comp.n_baseline_only == 1  # "a"
    assert comp.n_candidate_only == 1  # "c1"
    assert len(comp.tasks) == 3  # all ids
    assert [d.task_id for d in comp.tasks] == ["a", "c1", "shared"]


def test_compare_runs_pass_rate_counts_only_paired() -> None:
    # baseline: shared passes, a passes (but a is baseline-only -> excluded)
    base = _report(
        "b", [("a", 1.0, True, None), ("s", 0.4, False, None)]
    )
    cand = _report("c", [("s", 0.4, False, None), ("c1", 1.0, True, None)])
    comp = compare_runs(base, cand)
    assert comp.n_paired == 1
    assert comp.baseline_pass_rate == 0.0  # shared failed in baseline
    assert comp.candidate_pass_rate == 0.0  # shared failed in candidate
    assert comp.pass_rate_delta == 0.0


# ---------------------------------------------------------------------------
# Variance / CI edge cases
# ---------------------------------------------------------------------------


def test_zero_variance_collapses_ci() -> None:
    # Identical scores everywhere -> all deltas 0 -> std_delta 0.
    base = _report(
        "b", [("t1", 0.7, True, None), ("t2", 0.7, True, None)]
    )
    cand = _report(
        "c", [("t1", 0.7, True, None), ("t2", 0.7, True, None)]
    )
    comp = compare_runs(base, cand)
    assert comp.mean_delta == pytest.approx(0.0)
    assert comp.std_delta == pytest.approx(0.0)
    assert comp.variance_delta == pytest.approx(0.0)
    assert comp.ci_lower == pytest.approx(0.0)
    assert comp.ci_upper == pytest.approx(0.0)
    assert comp.cohens_d == 0.0
    assert comp.t_statistic == 0.0
    assert comp.p_value_two_tailed == 1.0


def test_single_paired_task_collapses_ci() -> None:
    base = _report("b", [("t1", 0.8, True, None)])
    cand = _report("c", [("t1", 0.2, False, None)])
    comp = compare_runs(base, cand)
    assert comp.n_paired == 1
    assert comp.mean_delta == pytest.approx(-0.6)
    assert comp.std_delta == 0.0  # ddof=1, n=1 -> variance 0
    assert comp.variance_delta == 0.0
    assert comp.ci_lower == pytest.approx(-0.6)
    assert comp.ci_upper == pytest.approx(-0.6)
    assert comp.t_statistic == 0.0
    assert comp.p_value_two_tailed == 1.0
    assert comp.regression_detected is False


def test_confidence_interval_function_no_paired() -> None:
    empty = RunReport(
        run_id="e", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    assert confidence_interval(empty, empty) == (0.0, 0.0)
    base = _report("b", [("a", 0.9, True, None)])
    cand = _report("c", [("c1", 0.9, True, None)])  # no overlap
    assert confidence_interval(base, cand) == (0.0, 0.0)


def test_confidence_level_affects_width() -> None:
    base, cand = _regress_pair()
    comp90 = compare_runs(base, cand, confidence_level=0.90)
    comp99 = compare_runs(base, cand, confidence_level=0.99)
    assert comp99.confidence_level == 0.99
    assert comp99.ci_lower < comp90.ci_lower
    assert comp99.ci_upper > comp90.ci_upper
    # Midpoint invariant: mean_delta == (ci_lower + ci_upper) / 2
    for comp in (comp90, comp99):
        mid = (comp.ci_lower + comp.ci_upper) / 2
        assert mid == pytest.approx(comp.mean_delta)
    # mean_delta always within [ci_lower, ci_upper]
    assert comp90.ci_lower <= comp90.mean_delta <= comp90.ci_upper


# ---------------------------------------------------------------------------
# Effect size & significance values
# ---------------------------------------------------------------------------


def test_strong_regression_significance() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert comp.mean_delta == pytest.approx(-0.66, abs=1e-9)
    assert comp.t_statistic < 0  # candidate worse
    assert comp.p_value_two_tailed < 0.05
    assert comp.p_value_greater > 0.5  # not greater; delta is negative
    # t-stat consistency: mean_delta / (std_delta / sqrt(n))
    se = comp.std_delta / math.sqrt(comp.n_paired)
    assert comp.t_statistic == pytest.approx(comp.mean_delta / se)
    # cohens_d consistency
    assert comp.cohens_d == pytest.approx(comp.mean_delta / comp.std_delta)


def test_strong_improvement_significance() -> None:
    base, cand = _improve_pair()
    comp = compare_runs(base, cand)
    assert comp.mean_delta > 0
    assert comp.t_statistic > 0
    assert comp.p_value_two_tailed < 0.05
    assert comp.p_value_greater < 0.05  # candidate significantly greater
    assert comp.p_value_greater >= 0.0
    assert comp.cohens_d > 0


def test_p_values_in_valid_range() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert 0.0 <= comp.p_value_two_tailed <= 1.0
    assert 0.0 <= comp.p_value_greater <= 1.0


def test_effect_size_function_matches_compare_runs() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert effect_size(base, cand) == pytest.approx(comp.cohens_d)


def test_paired_significance_test_dict() -> None:
    base, cand = _improve_pair()
    result = paired_significance_test(base, cand)
    assert set(result) == {
        "t_statistic",
        "df",
        "p_value_two_tailed",
        "p_value_greater",
        "mean_delta",
        "std_delta",
        "n_paired",
    }
    assert result["df"] == len(base.results) - 1
    assert result["n_paired"] == 5
    assert result["p_value_two_tailed"] < 0.05
    assert result["p_value_greater"] < 0.05
    assert result["mean_delta"] > 0


def test_paired_significance_test_no_data() -> None:
    empty = RunReport(
        run_id="e", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    result = paired_significance_test(empty, empty)
    assert result["n_paired"] == 0
    assert result["df"] == 0  # max(0, n-1)
    assert result["t_statistic"] == 0.0
    assert result["p_value_two_tailed"] == 1.0
    assert result["p_value_greater"] == 1.0
    assert result["mean_delta"] == 0.0
    assert result["std_delta"] == 0.0


# ---------------------------------------------------------------------------
# Helper function consistency
# ---------------------------------------------------------------------------


def test_pass_rate_delta_helper() -> None:
    base, cand = _improve_pair()
    comp = compare_runs(base, cand)
    assert pass_rate_delta(base, cand) == pytest.approx(comp.pass_rate_delta)
    # no paired -> 0.0
    empty = RunReport(
        run_id="e", suite_name="s", agent_name="a", started_at="2024-01-01T00:00:00Z"
    )
    assert pass_rate_delta(empty, empty) == 0.0


def test_mean_score_delta_helper() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert mean_score_delta(base, cand) == pytest.approx(comp.mean_score_delta)
    assert mean_score_delta(base, cand) == pytest.approx(comp.mean_delta)


def test_variance_delta_helper() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert variance_delta(base, cand) == pytest.approx(comp.variance_delta)
    assert variance_delta(base, cand) == pytest.approx(comp.std_delta ** 2)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def test_regression_detected_strong() -> None:
    base, cand = _regress_pair()
    comp = compare_runs(base, cand)
    assert comp.regression_detected is True
    assert comp.mean_delta < 0
    assert comp.p_value_two_tailed < 0.05
    assert detect_regression(base, cand) is True


def test_no_regression_on_improvement() -> None:
    base, cand = _improve_pair()
    comp = compare_runs(base, cand)
    assert comp.regression_detected is False
    assert comp.mean_delta > 0
    assert detect_regression(base, cand) is False


def test_no_regression_on_identical() -> None:
    base = _report(
        "b", [("t1", 0.9, True, None), ("t2", 0.5, False, None)]
    )
    cand = _report(
        "c", [("t1", 0.9, True, None), ("t2", 0.5, False, None)]
    )
    comp = compare_runs(base, cand)
    assert comp.mean_delta == pytest.approx(0.0)
    assert comp.regression_detected is False
    assert detect_regression(base, cand) is False


def test_no_regression_with_no_paired_tasks() -> None:
    base = _report("b", [("a", 0.9, True, None)])
    cand = _report("c", [("c1", 0.9, True, None)])
    comp = compare_runs(base, cand)
    assert comp.n_paired == 0
    assert comp.regression_detected is False
    assert detect_regression(base, cand) is False


def test_regression_threshold_plumbed() -> None:
    base, cand = _regress_pair()
    # p_two is tiny but strictly > 0, so threshold=0.0 disables regression.
    comp_loose = compare_runs(base, cand, regression_threshold=1.0)
    assert comp_loose.regression_detected is True
    comp_strict = compare_runs(base, cand, regression_threshold=0.0)
    assert comp_strict.regression_detected is False
    # detect_regression forwards the threshold.
    assert detect_regression(base, cand, threshold=1.0) is True
    assert detect_regression(base, cand, threshold=0.0) is False


def test_regression_requires_negative_mean() -> None:
    # Improvement (positive delta) never counts as a regression, even with a
    # very permissive threshold, because the mean_delta < 0 clause fails.
    base, cand = _improve_pair()
    assert compare_runs(base, cand, regression_threshold=1.0).regression_detected is False
    assert detect_regression(base, cand, threshold=1.0) is False


# ---------------------------------------------------------------------------
# RunComparison.build-style construction: tasks ordering
# ---------------------------------------------------------------------------


def test_runcomparison_tasks_ordered_by_task_id() -> None:
    base = _report(
        "b",
        [("zeta", 0.2, False, None), ("mid", 0.4, False, None)],
    )
    cand = _report(
        "c",
        [("zeta", 0.6, False, None), ("mid", 0.6, False, None), ("alpha", 0.9, True, None)],
    )
    comp = compare_runs(base, cand)
    ids = [d.task_id for d in comp.tasks]
    assert ids == sorted(ids)
    assert ids == ["alpha", "mid", "zeta"]
    # alpha is candidate-only: in_baseline False, baseline_score 0.0
    alpha = comp.tasks[0]
    assert alpha.in_baseline is False
    assert alpha.baseline_score == pytest.approx(0.0)
    assert alpha.candidate_score == pytest.approx(0.9)
    assert alpha.score_delta == pytest.approx(0.9)
