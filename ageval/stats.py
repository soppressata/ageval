from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ageval.core import RunReport, TaskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_value(r: TaskResult) -> float:
    """Return the score value for a result, treating missing scores as 0.0."""
    return r.score.value if r.score is not None else 0.0


def _passed(r: TaskResult) -> bool:
    """Return True iff the result passed (no agent error and score.passed)."""
    return r.ok


def _index_results(report: RunReport) -> dict[str, TaskResult]:
    """Build a task-id -> TaskResult map for a report."""
    return {r.task.id: r for r in report.results}


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _variance(xs: list[float], ddof: int = 0) -> float:
    """Population variance (ddof=0) or sample variance (ddof=1).

    Returns 0.0 when the denominator would be zero.
    """
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def _stdev(xs: list[float], ddof: int = 0) -> float:
    return math.sqrt(_variance(xs, ddof=ddof))


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _beta_contrib(a: float, b: float, x: float, errtol: float = 1e-12) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Based on the Lentz formulation (Numerical Recipes, Press et al.).
    Returns 0.0 for x <= 0 and 1.0 for x >= 1.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Log of the prefactor so we don't overflow for large a, b.
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Lentz continued fraction for the tail ratio.
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, 1000):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < errtol:
            break

    return front * h * a


def _t_cdf(t_val: float, df: float) -> float:
    """Cumulative distribution function of Student's t with df degrees of freedom.

    Uses the regularized incomplete beta function:
        F(t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)   if t >= 0
        F(t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)         if t < 0
    """
    if df <= 0:
        return _normal_cdf(t_val)
    x = df / (df + t_val * t_val)
    ib = _beta_contrib(df / 2.0, 0.5, x)
    if t_val >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskDelta:
    """Paired comparison of a single task across two runs.

    Attributes:
        task_id: The shared task identifier.
        baseline_score: Score value in the baseline run (0.0 if missing/errored).
        candidate_score: Score value in the candidate run (0.0 if missing/errored.
        baseline_passed: Whether the baseline run passed this task.
        candidate_passed: Whether the candidate run passed this task.
        score_delta: candidate_score - baseline_score.
        in_baseline: True if the task appeared in the baseline run.
        in_candidate: True if the task appeared in the candidate run.
    """

    task_id: str
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    baseline_passed: bool = False
    candidate_passed: bool = False
    score_delta: float = 0.0
    in_baseline: bool = True
    in_candidate: bool = True


@dataclass
class RunComparison:
    """Aggregate statistics comparing a candidate run against a baseline run.

    Attributes:
        baseline_run_id: Run ID of the baseline.
        candidate_run_id: Run ID of the candidate.
        n_paired: Number of tasks present in both runs.
        n_baseline_only: Tasks only in the baseline.
        n_candidate_only: Tasks only in the candidate.
        baseline_pass_rate: Pass rate of the baseline over paired tasks.
        candidate_pass_rate: Pass rate of the candidate over paired tasks.
        pass_rate_delta: candidate_pass_rate - baseline_pass_rate.
        baseline_mean_score: Mean score of the baseline over paired tasks.
        candidate_mean_score: Mean score of the candidate over paired tasks.
        mean_score_delta: candidate_mean_score - baseline_mean_score.
        mean_delta: Mean of per-task score differences (candidate - baseline).
        variance_delta: Sample variance of per-task score differences.
        std_delta: Sample standard deviation of per-task score differences.
        ci_lower: Lower bound of the confidence interval for mean_delta.
        ci_upper: Upper bound of the confidence interval for mean_delta.
        confidence_level: Confidence level used for the interval (e.g. 0.95).
        t_statistic: Paired t-statistic for mean_delta.
        p_value_two_tailed: Two-tailed p-value for the paired t-test.
        p_value_greater: One-tailed p-value for H1: mean_delta > 0.
        cohens_d: Paired effect size (mean_delta / std_delta).
        regression_detected: True if a significant negative effect is found.
        tasks: Per-task paired deltas, ordered by task_id.
    """

    baseline_run_id: str = ""
    candidate_run_id: str = ""
    n_paired: int = 0
    n_baseline_only: int = 0
    n_candidate_only: int = 0
    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    pass_rate_delta: float = 0.0
    baseline_mean_score: float = 0.0
    candidate_mean_score: float = 0.0
    mean_score_delta: float = 0.0
    mean_delta: float = 0.0
    variance_delta: float = 0.0
    std_delta: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    confidence_level: float = 0.95
    t_statistic: float = 0.0
    p_value_two_tailed: float = 1.0
    p_value_greater: float = 1.0
    cohens_d: float = 0.0
    regression_detected: bool = False
    tasks: list[TaskDelta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def paired_deltas(
    baseline: RunReport, candidate: RunReport
) -> list[TaskDelta]:
    """Compute per-task score deltas for all tasks appearing in either report.

    Tasks present in both reports are paired by task ID. Tasks appearing in only
    one run report a delta against an implicit zero score for the missing side.

    Results are returned sorted by task_id for deterministic output.
    """
    base_idx = _index_results(baseline)
    cand_idx = _index_results(candidate)

    all_ids = sorted(set(base_idx) | set(cand_idx))
    deltas: list[TaskDelta] = []

    for tid in all_ids:
        in_b = tid in base_idx
        in_c = tid in cand_idx
        br = base_idx[tid] if in_b else None
        cr = cand_idx[tid] if in_c else None

        b_score = _score_value(br) if br is not None else 0.0
        c_score = _score_value(cr) if cr is not None else 0.0
        b_pass = _passed(br) if br is not None else False
        c_pass = _passed(cr) if cr is not None else False

        deltas.append(
            TaskDelta(
                task_id=tid,
                baseline_score=b_score,
                candidate_score=c_score,
                baseline_passed=b_pass,
                candidate_passed=c_pass,
                score_delta=c_score - b_score,
                in_baseline=in_b,
                in_candidate=in_c,
            )
        )

    return deltas


def _paired_only_deltas(deltas: list[TaskDelta]) -> list[TaskDelta]:
    """Return only deltas where the task appears in both runs."""
    return [d for d in deltas if d.in_baseline and d.in_candidate]


def compare_runs(
    baseline: RunReport,
    candidate: RunReport,
    confidence_level: float = 0.95,
    regression_threshold: float = 0.05,
) -> RunComparison:
    """Compare a candidate RunReport against a baseline RunReport.

    Uses only tasks present in *both* runs for paired statistical tests
    (significance, effect size, confidence interval). Tasks unique to one run
    are reported in the counts ``n_baseline_only`` / ``n_candidate_only`` but do
    not contribute to the statistical tests.

    The paired t-test evaluates H0: mean(score_delta) == 0 against the two-sided
    alternative H1: mean(score_delta) != 0, plus a one-sided "greater" test
    H1: mean(score_delta) > 0 (candidate beats baseline).

    Regression is flagged when the two-tailed p-value falls below
    ``regression_threshold`` AND ``mean_delta`` is negative (candidate worse).

    Args:
        baseline: The reference run.
        candidate: The run under evaluation.
        confidence_level: Confidence level for the interval (default 0.95).
        regression_threshold: Significance threshold for regression detection.

    Returns:
        A :class:`RunComparison` with all fields populated.
    """
    all_deltas = paired_deltas(baseline, candidate)
    paired = _paired_only_deltas(all_deltas)

    n_paired = len(paired)
    n_baseline_only = sum(1 for d in all_deltas if d.in_baseline and not d.in_candidate)
    n_candidate_only = sum(1 for d in all_deltas if d.in_candidate and not d.in_baseline)

    # Pass rates over paired tasks.
    if n_paired == 0:
        b_pass_rate = 0.0
        c_pass_rate = 0.0
    else:
        b_pass_rate = sum(1 for d in paired if d.baseline_passed) / n_paired
        c_pass_rate = sum(1 for d in paired if d.candidate_passed) / n_paired

    b_mean = _mean([d.baseline_score for d in paired])
    c_mean = _mean([d.candidate_score for d in paired])

    diffs = [d.score_delta for d in paired]
    mean_delta = _mean(diffs)
    var_delta = _variance(diffs, ddof=1)
    std_delta = math.sqrt(var_delta)

    # Confidence interval for the mean difference.
    if n_paired >= 2 and std_delta > 0.0:
        se = std_delta / math.sqrt(n_paired)
        alpha = 1.0 - confidence_level
        # Two-sided critical value using the t-distribution.
        # p = 1 - alpha/2  ->  t_cdf(t) = 1 - alpha/2.
        # Use a normal approximation for the critical value, refined by
        # one Newton step toward the exact t-quantile.
        z = _normal_quantile(1.0 - alpha / 2.0)
        # Refine: convert normal z to approximate t via expansion.
        df = n_paired - 1.0
        t_crit = _t_quantile(1.0 - alpha / 2.0, df, hint=z)
        ci_lower = mean_delta - t_crit * se
        ci_upper = mean_delta + t_crit * se
    else:
        ci_lower = mean_delta
        ci_upper = mean_delta

    # Paired t-test.
    if n_paired >= 2 and std_delta > 0.0:
        se = std_delta / math.sqrt(n_paired)
        t_stat = mean_delta / se
        df = n_paired - 1.0
        p_two = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
        p_greater = 1.0 - _t_cdf(t_stat, df)
    else:
        t_stat = 0.0
        p_two = 1.0
        p_greater = 1.0

    # Paired effect size (Cohen's d for paired data).
    if std_delta > 0.0:
        cohens_d = mean_delta / std_delta
    else:
        cohens_d = 0.0

    regression = (p_two < regression_threshold) and (mean_delta < 0.0)

    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        n_paired=n_paired,
        n_baseline_only=n_baseline_only,
        n_candidate_only=n_candidate_only,
        baseline_pass_rate=b_pass_rate,
        candidate_pass_rate=c_pass_rate,
        pass_rate_delta=c_pass_rate - b_pass_rate,
        baseline_mean_score=b_mean,
        candidate_mean_score=c_mean,
        mean_score_delta=c_mean - b_mean,
        mean_delta=mean_delta,
        variance_delta=var_delta,
        std_delta=std_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        t_statistic=t_stat,
        p_value_two_tailed=p_two,
        p_value_greater=p_greater,
        cohens_d=cohens_d,
        regression_detected=regression,
        tasks=all_deltas,
    )


def pass_rate_delta(baseline: RunReport, candidate: RunReport) -> float:
    """Difference in pass rate (candidate - baseline) over paired tasks.

    Returns 0.0 when there are no paired tasks.
    """
    paired = _paired_only_deltas(paired_deltas(baseline, candidate))
    if not paired:
        return 0.0
    b = sum(1 for d in paired if d.baseline_passed) / len(paired)
    c = sum(1 for d in paired if d.candidate_passed) / len(paired)
    return c - b


def mean_score_delta(baseline: RunReport, candidate: RunReport) -> float:
    """Difference in mean score (candidate - baseline) over paired tasks.

    Returns 0.0 when there are no paired tasks.
    """
    paired = _paired_only_deltas(paired_deltas(baseline, candidate))
    if not paired:
        return 0.0
    return _mean([d.candidate_score for d in paired]) - _mean(
        [d.baseline_score for d in paired]
    )


def variance_delta(baseline: RunReport, candidate: RunReport) -> float:
    """Sample variance of per-task score differences over paired tasks.

    Returns 0.0 when fewer than two tasks are paired.
    """
    paired = _paired_only_deltas(paired_deltas(baseline, candidate))
    return _variance([d.score_delta for d in paired], ddof=1)


def confidence_interval(
    baseline: RunReport,
    candidate: RunReport,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for the mean score delta over paired tasks.

    Returns ``(0.0, 0.0)`` when there are no paired tasks. When there is exactly
    one paired task (or zero variance) the interval collapses to the point
    estimate.
    """
    comp = compare_runs(baseline, candidate, confidence_level=confidence_level)
    return (comp.ci_lower, comp.ci_upper)


def paired_significance_test(
    baseline: RunReport,
    candidate: RunReport,
) -> dict[str, Any]:
    """Run a paired t-test on per-task score deltas.

    Tests H0: mean(score_delta) == 0 against H1: mean != 0 (two-sided) and
    H1: mean > 0 (candidate better).

    Returns a dict with keys ``t_statistic``, ``df``, ``p_value_two_tailed``,
    ``p_value_greater``, ``mean_delta``, ``std_delta``, ``n_paired``.
    Values are ``0.0`` / ``1.0`` defaults when the test cannot be performed.
    """
    paired = _paired_only_deltas(paired_deltas(baseline, candidate))
    n = len(paired)
    diffs = [d.score_delta for d in paired]
    md = _mean(diffs)
    sd = _stdev(diffs, ddof=1)

    if n < 2 or sd == 0.0:
        return {
            "t_statistic": 0.0,
            "df": max(0, n - 1),
            "p_value_two_tailed": 1.0,
            "p_value_greater": 1.0,
            "mean_delta": md,
            "std_delta": sd,
            "n_paired": n,
        }

    se = sd / math.sqrt(n)
    t_stat = md / se
    df = float(n - 1)
    p_two = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    p_greater = 1.0 - _t_cdf(t_stat, df)

    return {
        "t_statistic": t_stat,
        "df": n - 1,
        "p_value_two_tailed": p_two,
        "p_value_greater": p_greater,
        "mean_delta": md,
        "std_delta": sd,
        "n_paired": n,
    }


def effect_size(
    baseline: RunReport, candidate: RunReport
) -> float:
    """Paired Cohen's d for score deltas over shared tasks.

    Defined as mean_delta / std_delta of the paired differences. Returns 0.0
    when there are fewer than two paired tasks or when the standard deviation is
    zero (no variance means the effect size is undefined; we return 0.0 rather
    than raise).
    """
    paired = _paired_only_deltas(paired_deltas(baseline, candidate))
    if len(paired) < 2:
        return 0.0
    diffs = [d.score_delta for d in paired]
    sd = _stdev(diffs, ddof=1)
    if sd == 0.0:
        return 0.0
    return _mean(diffs) / sd


def detect_regression(
    baseline: RunReport,
    candidate: RunReport,
    threshold: float = 0.05,
) -> bool:
    """Return True if the candidate is significantly worse than the baseline.

    Regression is defined as a two-tailed paired t-test p-value below
    ``threshold`` with a negative mean score delta.
    """
    comp = compare_runs(baseline, candidate, regression_threshold=threshold)
    return comp.regression_detected


# ---------------------------------------------------------------------------
# Distribution quantile helpers (private)
# ---------------------------------------------------------------------------


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (probit) via the Acklam approximation.

    Accurate to ~1e-9 in the central region. Clamps ``p`` to (0, 1).
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    # Peter Acklam's approximation.
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (
                (((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]
            )
            * r
            + a[5]
        ) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))


def _t_quantile(p: float, df: float, hint: float | None = None, tol: float = 1e-10) -> float:
    """Inverse t CDF (quantile function) via Newton-Raphson on _t_cdf.

    ``hint`` is an initial guess (defaults to the normal quantile).
    Converges rapidly because the t-distribution is smooth and unimodal.
    """
    if df <= 0:
        return _normal_quantile(p)
    x = hint if hint is not None else _normal_quantile(p)

    for _ in range(100):
        f = _t_cdf(x, df) - p
        # PDF of t at x.
        u = df / (df + x * x)
        # Ratio of gamma functions via lgamma.
        pdf = (
            math.exp(
                math.lgamma((df + 1) / 2.0)
                - math.lgamma(df / 2.0)
                - 0.5 * math.log(df * math.pi)
                + (df / 2.0) * math.log(u)
            )
        )
        if pdf <= 0.0:
            break
        step = f / pdf
        x -= step
        if abs(step) < tol * (1.0 + abs(x)):
            break
    return x
