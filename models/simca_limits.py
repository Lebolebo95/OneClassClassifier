"""Statistical limits for SIMCA distance statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import erfinv


@dataclass(frozen=True)
class DistanceLimits:
    """Acceptance limits for SIMCA distance statistics.

    q, t2     : individual distance limits (nan when not applicable).
    combined  : combined acceptance frontier.
    outlier   : outlier frontier (DD only, nan otherwise).
    """
    q: float
    t2: float
    combined: float
    outlier: float


def q_limit(
    q: np.ndarray,
    residual_eigenvalues: np.ndarray,
    *,
    confidence: float,
    method: str,
    eps: float = 1e-12,
) -> float:
    """Return the Q residual-distance acceptance limit."""

    q = np.asarray(q, dtype=float)
    method = str(method).lower()

    if method == "percentile":
        return float(np.quantile(q, confidence))

    residual_eigenvalues = np.asarray(residual_eigenvalues, dtype=float)

    if method == "jm":
        return _q_limit_jackson_mudholkar(residual_eigenvalues, confidence, eps)

    theta1 = float(np.sum(residual_eigenvalues))
    theta2 = float(np.sum(residual_eigenvalues ** 2))

    if method == "chi2box":
        theta1 = max(theta1, eps)
        theta2 = max(theta2, eps)
        return float((theta2 / theta1) * stats.chi2.ppf(confidence, theta1 ** 2 / theta2))

    if method in {"chi2", "chi2pom"}:
        mean = max(float(np.mean(q)), eps)
        variance = max(float(np.var(q, ddof=1)), eps)
        dof = 2.0 * mean ** 2 / variance
        if method == "chi2pom":
            dof = max(round(dof), 1)
        return float(mean * stats.chi2.ppf(confidence, dof) / dof)

    raise ValueError(f"q_threshold must be 'jm', 'chi2box', 'chi2', 'chi2pom' or 'percentile', got {method!r}")


def t2_limit(
    t2: np.ndarray,
    *,
    n_components: int,
    n_samples: int,
    confidence: float,
    method: str,
    eps: float = 1e-12,
) -> float:
    """Return the Hotelling T2 score-distance acceptance limit."""

    t2 = np.asarray(t2, dtype=float)
    method = str(method).lower()

    if method == "percentile":
        return float(np.quantile(t2, confidence, method="interpolated_inverted_cdf"))

    if method == "chi2":
        return float(stats.chi2.ppf(confidence, n_components))

    if method in {"fdist", "fdistrig"}:
        return _t2_limit_f(n_components, n_samples, confidence, rigorous=(method == "fdistrig"))

    if method == "chi2pom":
        mean = max(float(np.mean(t2)), eps)
        variance = max(float(np.var(t2, ddof=1)), eps)
        dof = max(round(2.0 * mean ** 2 / variance), 1)
        return float(mean * stats.chi2.ppf(confidence, dof) / dof)

    raise ValueError(f"t2_threshold must be 'fdist', 'fdistrig', 'chi2', 'chi2pom' or 'percentile', got {method!r}")


def combined_limit(
    method: str,
    *,
    n_components: int,
    q_lim: float,
    t2_lim: float,
    residual_eigenvalues: np.ndarray,
    confidence: float,
    eps: float = 1e-12,
) -> float:
    """Return the combined acceptance frontier for non-DD methods."""

    method = str(method).lower()

    if method == "sim":
        return 1.0
    if method == "alt":
        return float(np.sqrt(2.0))

    if method in {"combined", "ci"}:
        residual_eigenvalues = np.asarray(residual_eigenvalues, dtype=float)
        theta1 = float(np.sum(residual_eigenvalues))
        theta2 = float(np.sum(residual_eigenvalues ** 2))
        tr1 = (n_components / t2_lim) + (theta1 / q_lim)
        tr2 = (n_components / t2_lim ** 2) + (theta2 / q_lim ** 2)
        _check_positive("combined tr1", tr1, eps)
        _check_positive("combined tr2", tr2, eps)
        scale = tr2 / tr1
        dof = tr1 ** 2 / tr2
        return float(scale * stats.chi2.ppf(confidence, dof))

    raise ValueError(f"unsupported SIMCA method: {method!r}")


# --- DD-SIMCA ---

@dataclass(frozen=True)
class DDParameters:
    """Moment-matched chi-square parameters for DD-SIMCA."""
    q_center: float
    t2_center: float
    q_variance: float
    t2_variance: float
    q_dof: float
    t2_dof: float


def dd_parameters(q: np.ndarray, t2: np.ndarray, *, eps: float = 1e-12) -> DDParameters:
    """Estimate DD-SIMCA parameters from training Q and T2 distances."""

    q = np.asarray(q, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    q_center = max(float(np.mean(q)), eps)
    t2_center = max(float(np.mean(t2)), eps)
    q_var = max(float(np.var(q, ddof=1)), eps)
    t2_var = max(float(np.var(t2, ddof=1)), eps)
    return DDParameters(
        q_center=q_center,
        t2_center=t2_center,
        q_variance=q_var,
        t2_variance=t2_var,
        q_dof=2.0 * q_center ** 2 / q_var,
        t2_dof=2.0 * t2_center ** 2 / t2_var,
    )


def dd_alpha_limit(params: DDParameters, alpha: float) -> float:
    """Return the DD-SIMCA acceptance frontier at significance level alpha."""
    return float(stats.chi2.ppf(1.0 - alpha, params.q_dof + params.t2_dof))


def dd_outlier_limit(params: DDParameters, gamma: float, n_samples: int) -> float:
    """Return the DD-SIMCA outlier frontier with Šidák correction for n_samples tests."""
    per_comparison = (1.0 - gamma) ** (1.0 / n_samples)
    return float(stats.chi2.ppf(per_comparison, params.q_dof + params.t2_dof))


# --- private helpers ---

def _q_limit_jackson_mudholkar(
    residual_eigenvalues: np.ndarray,
    confidence: float,
    eps: float,
) -> float:
    theta1 = float(np.sum(residual_eigenvalues))
    theta2 = float(np.sum(residual_eigenvalues ** 2))
    theta3 = float(np.sum(residual_eigenvalues ** 3))
    if theta1 <= eps:
        return 0.0
    theta2 = max(theta2, eps)
    h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 ** 2)
    h0 = max(h0, 0.001)
    ca = np.sqrt(2.0) * erfinv(2.0 * confidence - 1.0)
    h1 = ca * np.sqrt(2.0 * theta2 * h0 ** 2) / theta1
    h2 = theta2 * h0 * (h0 - 1.0) / theta1 ** 2
    return float(theta1 * (1.0 + h1 + h2) ** (1.0 / h0))


def _t2_limit_f(
    n_components: int,
    n_samples: int,
    confidence: float,
    *,
    rigorous: bool = False,
) -> float:
    if n_samples <= n_components:
        raise ValueError("F-distribution T2 limit requires n_samples > n_components")
    f_value = stats.f.ppf(confidence, n_components, n_samples - n_components)
    if rigorous:
        return float(
            n_components * (n_samples ** 2 - 1.0) / (n_samples * (n_samples - n_components)) * f_value
        )
    return float(n_components * (n_samples - 1.0) / (n_samples - n_components) * f_value)


def _check_positive(name: str, value: float, eps: float) -> None:
    if not np.isfinite(value) or value <= eps:
        raise ValueError(f"{name} must be positive and finite, got {value}")
