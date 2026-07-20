"""SIMCA one-class class modeling estimator."""

from __future__ import annotations

import warnings
from typing import Literal, get_args

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import OneClassMixin
from .simca_plots import SIMCAPlotMixin
from .simca_limits import (
    DDParameters,
    DistanceLimits,
    dd_alpha_limit,
    dd_outlier_limit,
    dd_parameters,
    combined_limit,
    q_limit,
    t2_limit,
)

Method = Literal["sim", "alt", "combined", "ci", "dd"]
QThreshold = Literal["jm", "chi2box", "chi2", "chi2pom", "percentile"]
T2Threshold = Literal["fdist", "fdistrig", "chi2", "chi2pom", "percentile"]

# Kept as sets (not just the Literal aliases) because .lower() input needs a
# runtime check — get_args() reuses the Literal values so both stay in sync.
_VALID_METHODS = set(get_args(Method))
_VALID_Q_THRESHOLDS = set(get_args(QThreshold))
_VALID_T2_THRESHOLDS = set(get_args(T2Threshold))


class SIMCA(SIMCAPlotMixin, OneClassMixin, BaseEstimator):
    """SIMCA one-class class model.

    Fits a PCA model on target-class samples and classifies new samples
    based on their Q (residual) and T2 (score) distances.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components.
    method : {'sim', 'alt', 'combined', 'ci', 'dd'}, default='alt'
        Decision rule for combining Q and T2 distances.
        'sim'      accepts if both Q <= q_limit AND T2 <= t2_limit.
        'alt'      accepts if sqrt((Q/q_lim)^2 + (T2/t2_lim)^2) <= sqrt(2).
        'combined' / 'ci' uses a chi2-based combined statistic.
        'dd'       uses DD-SIMCA moment-matched chi2 distances.
    alpha : float, default=0.05
        Significance level for the acceptance limit.
    gamma : float, default=0.01
        Significance level for the DD-SIMCA outlier limit.
    q_threshold : {'jm', 'chi2box', 'chi2', 'chi2pom', 'percentile'}, default='jm'
        Method for estimating the Q limit.
    t2_threshold : {'fdist', 'fdistrig', 'chi2', 'chi2pom', 'percentile'}, default='fdist'
        Method for estimating the T2 limit.
    center : bool, default=True
        Whether to mean-center before PCA. Should be True in nearly all cases.
    eps : float, default=1e-12
        Small regularisation constant to avoid division by zero.
    target_class : label, default=None
        Target class label. Required when y has more than one class.
    preprocessor : transformer or None, default=None
        Applied only to target-class samples before PCA. Must implement fit/transform.
    tuning : {'rigorous', 'compliant'}, default='rigorous'
        Scoring strategy for GridSearchCV — see OneClassMixin.score() for
        exactly what each mode optimises.
    positive_label : int, default=1
        Label for accepted samples.
    negative_label : int, default=-1
        Label for rejected samples.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        method: Method = "alt",
        alpha: float = 0.05,
        gamma: float = 0.01,
        q_threshold: QThreshold = "jm",
        t2_threshold: T2Threshold = "fdist",
        center: bool = True,
        eps: float = 1e-12,
        target_class=None,
        preprocessor=None,
        tuning: str = "rigorous",
        positive_label: int = 1,
        negative_label: int = -1,
    ) -> None:
        self.n_components = n_components
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.q_threshold = q_threshold
        self.t2_threshold = t2_threshold
        self.center = center
        self.eps = eps
        self.target_class = target_class
        self.preprocessor = preprocessor
        self.tuning = tuning
        self.positive_label = positive_label
        self.negative_label = negative_label

    # --- OneClassMixin hooks ---

    def _fit_one_class_model(self, X_target: np.ndarray) -> None:
        n_samples, n_features = X_target.shape
        self._validate_simca_params(n_samples, n_features)

        method = self.method.lower()

        # PCA decomposition
        if self.center:
            pca = PCA(n_components=None, svd_solver="auto")
            pca.fit(X_target)
            self.mean_ = pca.mean_
            full_variance = pca.explained_variance_
            full_ratio = pca.explained_variance_ratio_
            full_components = pca.components_
        else:
            self.mean_ = np.zeros(n_features)
            _, sv, full_components = np.linalg.svd(X_target, full_matrices=False)
            full_variance = sv ** 2 / max(n_samples - 1, 1)
            full_ratio = full_variance / np.sum(full_variance)

        self.components_ = full_components[: self.n_components]       # (n_components, n_features)
        self.loadings_ = self.components_.T                            # (n_features, n_components)
        self.explained_variance_ = full_variance[: self.n_components]
        self.explained_variance_ratio_ = full_ratio[: self.n_components]
        self.residual_eigenvalues_ = full_variance[self.n_components :]

        # Training distances
        scores, q, t2 = self._decompose(X_target)
        self.scores_ = scores
        self.training_q_ = q
        self.training_t2_ = t2
        self.n_samples_fit_ = n_samples

        # Acceptance limits
        self.thresholds_ = self._estimate_limits(q, t2, method)

    def _predict_details_one_class(self, X: np.ndarray) -> dict:
        method = self.method.lower()
        _, q, t2 = self._decompose(X)
        statistic = self._decision_statistic(q, t2, method)
        accepted = self._is_accepted(q, t2, statistic, method)
        decision = self._decision_margin(q, t2, statistic, method)

        return {
            "accepted": accepted,
            "decision": decision,
            "q": q,
            "t2": t2,
            "statistic": statistic,
            "q_limit": self.thresholds_.q,
            "t2_limit": self.thresholds_.t2,
            "alpha_limit": self.thresholds_.combined,
            "gamma_limit": self.thresholds_.outlier,
            "normalized_q": self._norm_q(q, method),
            "normalized_t2": self._norm_t2(t2, method),
            "status": self._status(statistic, accepted, method),
        }

    # --- SIMCA-specific public methods ---

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project samples onto the PCA subspace. Returns scores matrix."""
        X = self._transform_input(X)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Reconstruct samples from PCA scores."""
        scores = check_array(scores, ensure_2d=True, dtype=float)
        if scores.shape[1] != self.n_components:
            raise ValueError(
                f"scores must have {self.n_components} columns, got {scores.shape[1]}"
            )
        X_model = scores @ self.components_ + self.mean_
        if self.preprocessor_ is not None and hasattr(self.preprocessor_, "inverse_transform"):
            return self.preprocessor_.inverse_transform(X_model)
        return X_model

    def residual_contributions(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Return signed per-variable contributions to Q.

        sign(residual) * residual**2, so sum(abs(contributions)) over
        variables equals Q exactly for each sample (reference convention).
        """
        X = self._transform_input(X)
        n = self.n_components if n_components is None else int(n_components)
        raw = self._residuals(X, n)
        return np.sign(raw) * raw**2

    def score_contributions(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        *,
        absolute: bool = False,
    ) -> np.ndarray:
        """Return signed per-variable contributions to T2.

        sign(raw) * raw**2, so sum(abs(contributions)) over variables
        equals T2 exactly for each sample (reference convention).
        """
        X = self._transform_input(X)
        n = self.n_components if n_components is None else int(n_components)
        raw = self._t2_contributions(X, n)
        contrib = np.sign(raw) * raw**2
        return np.abs(contrib) if absolute else contrib

    # --- limit properties ---

    @property
    def q_limit(self) -> float:
        check_is_fitted(self, "thresholds_")
        return self.thresholds_.q

    @property
    def t2_limit(self) -> float:
        check_is_fitted(self, "thresholds_")
        return self.thresholds_.t2

    @property
    def alpha_limit(self) -> float:
        check_is_fitted(self, "thresholds_")
        return self.thresholds_.combined

    @property
    def gamma_limit(self) -> float:
        check_is_fitted(self, "thresholds_")
        return self.thresholds_.outlier

    # --- private helpers ---

    def _decompose(self, X: np.ndarray):
        """Return (scores, q, t2) for X.

        Q is the squared residual left outside the PCA subspace (how far a
        sample sits from the model). T2 is the Mahalanobis distance of the
        scores inside the subspace (how extreme a sample is among in-model
        variation). Together they're the two ways a sample can fail to look
        like the target class.
        """
        centered = X - self.mean_
        scores = centered @ self.components_.T
        eigs = np.maximum(self.explained_variance_, self.eps)
        residuals = centered - scores @ self.components_
        q = np.sum(residuals ** 2, axis=1)
        t2 = np.sum(scores ** 2 / eigs, axis=1)
        return scores, q, t2

    def _residuals(self, X: np.ndarray, n_components: int) -> np.ndarray:
        components = self.components_[:n_components]
        centered = X - self.mean_
        scores = centered @ components.T
        return centered - scores @ components

    def _t2_contributions(self, X: np.ndarray, n_components: int) -> np.ndarray:
        components = self.components_[:n_components]
        eigs = np.maximum(self.explained_variance_[:n_components], self.eps)
        centered = X - self.mean_
        scores = centered @ components.T
        return (scores / np.sqrt(eigs)) @ components

    def _estimate_limits(self, q: np.ndarray, t2: np.ndarray, method: str) -> DistanceLimits:
        confidence = 1.0 - self.alpha

        if method == "dd":
            params = self._dd_params(q, t2)
            self.dd_params_ = params
            effective_gamma = self.gamma
            if self.gamma >= self.alpha:
                # outlier limit must sit outside the class boundary, or every
                # rejected sample would also count as an "extreme" one
                effective_gamma = self.alpha / 2.0
                warnings.warn(
                    f"gamma ({self.gamma}) >= alpha ({self.alpha}): "
                    f"using gamma={effective_gamma} for the outlier limit.",
                    UserWarning,
                    stacklevel=4,
                )
            return DistanceLimits(
                q=np.nan,
                t2=np.nan,
                combined=dd_alpha_limit(params, self.alpha),
                outlier=dd_outlier_limit(params, effective_gamma, self.n_samples_fit_),
            )

        q_lim = q_limit(
            q,
            self.residual_eigenvalues_,
            confidence=confidence,
            method=self.q_threshold,
            eps=self.eps,
        )
        t2_lim = t2_limit(
            t2,
            n_components=self.n_components,
            n_samples=self.n_samples_fit_,
            confidence=confidence,
            method=self.t2_threshold,
            eps=self.eps,
        )
        comb = combined_limit(
            method,
            n_components=self.n_components,
            q_lim=q_lim,
            t2_lim=t2_lim,
            residual_eigenvalues=self.residual_eigenvalues_,
            confidence=confidence,
            eps=self.eps,
        )
        return DistanceLimits(q=q_lim, t2=t2_lim, combined=comb, outlier=np.nan)

    def _dd_params(self, q: np.ndarray, t2: np.ndarray) -> DDParameters:
        params = dd_parameters(q, t2, eps=self.eps)
        if self.q_threshold == "chi2pom" or self.t2_threshold == "chi2pom":
            return DDParameters(
                q_center=params.q_center,
                t2_center=params.t2_center,
                q_variance=params.q_variance,
                t2_variance=params.t2_variance,
                q_dof=max(round(params.q_dof), 1),
                t2_dof=max(round(params.t2_dof), 1),
            )
        return params

    def _decision_statistic(self, q, t2, method: str) -> np.ndarray:
        if method == "sim":
            # rectangular boundary: reject if either distance alone is too large
            return np.maximum(q / self.thresholds_.q, t2 / self.thresholds_.t2)
        if method == "alt":
            # elliptical boundary: the two distances can partially offset each other
            return np.sqrt((q / self.thresholds_.q) ** 2 + (t2 / self.thresholds_.t2) ** 2)
        if method in {"combined", "ci"}:
            return q / self.thresholds_.q + t2 / self.thresholds_.t2
        # dd: moment-matched chi-square scaling (dof/center from _dd_params) instead
        # of the alpha/gamma percentile limits used by the other methods
        return (
            self.dd_params_.q_dof * q / self.dd_params_.q_center
            + self.dd_params_.t2_dof * t2 / self.dd_params_.t2_center
        )

    def _is_accepted(self, q, t2, statistic, method: str) -> np.ndarray:
        if method == "sim":
            return (q <= self.thresholds_.q) & (t2 <= self.thresholds_.t2)
        return statistic <= self.thresholds_.combined

    def _decision_margin(self, q, t2, statistic, method: str) -> np.ndarray:
        if method == "sim":
            return np.minimum(self.thresholds_.q - q, self.thresholds_.t2 - t2)
        return self.thresholds_.combined - statistic

    def _norm_q(self, q, method: str) -> np.ndarray:
        if method == "dd":
            return self.dd_params_.q_dof * q / self.dd_params_.q_center
        return q / self.thresholds_.q

    def _norm_t2(self, t2, method: str) -> np.ndarray:
        if method == "dd":
            return self.dd_params_.t2_dof * t2 / self.dd_params_.t2_center
        return t2 / self.thresholds_.t2

    def _status(self, statistic, accepted, method: str) -> np.ndarray:
        if method != "dd":
            return np.where(accepted, "accepted", "rejected")
        # DD-SIMCA has a third bucket between the alpha (class) and gamma
        # (outlier) limits: "extreme" members that still belong to the class
        # but are unusual enough to be worth a second look.
        return np.select(
            [accepted, statistic <= self.thresholds_.outlier],
            ["accepted", "extreme"],
            default="outlier",
        )

    def _validate_simca_params(self, n_samples: int, n_features: int) -> None:
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if self.n_components >= min(n_samples, n_features):
            raise ValueError(
                f"n_components ({self.n_components}) must be < min(n_samples, n_features) "
                f"= min({n_samples}, {n_features}) = {min(n_samples, n_features)}"
            )
        if n_samples < 3:
            raise ValueError("SIMCA requires at least 3 target samples")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.method.lower() not in _VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}")
        if self.q_threshold.lower() not in _VALID_Q_THRESHOLDS:
            raise ValueError(f"q_threshold must be one of {sorted(_VALID_Q_THRESHOLDS)}")
        if self.t2_threshold.lower() not in _VALID_T2_THRESHOLDS:
            raise ValueError(f"t2_threshold must be one of {sorted(_VALID_T2_THRESHOLDS)}")
