"""Base mixin for one-class class modeling estimators."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .metrics import one_class_metrics


class OneClassMixin:
    """Mixin for one-class class modeling estimators.

    Handles target selection, target-only preprocessing, prediction and scoring.
    Subclasses must implement _fit_one_class_model and _predict_details_one_class.

    Parameters
    ----------
    target_class : label, default=None
        The class label to model. Required when y has more than one class.
    preprocessor : transformer or None, default=None
        Fitted only on target-class samples. Must implement fit and transform.
    tuning : {'rigorous', 'compliant'}, default='rigorous'
        Scoring strategy used by GridSearchCV.
        'rigorous' -> score() rewards configurations whose sensitivity on
                      target samples clears the nominal level (1 - self.alpha)
                      as consistently as possible across CV folds, then
                      prefers the most complex model (highest
                      self.n_components) among equally consistent ones.
                      Requires the subclass to expose `alpha` and
                      `n_components` attributes.
        'compliant' -> score() returns efficiency = sqrt(sensitivity * specificity).
                       Sensitivity is computed on target samples in X, specificity
                       on non-target samples in X — both straight from whatever
                       is passed to score(), not from anything stored at fit time.
    positive_label : int, default=1
        Label assigned to accepted (target) samples by predict().
    negative_label : int, default=-1
        Label assigned to rejected samples by predict().
    """

    def fit(self, X, y=None):
        """Fit the one-class model on target samples only.

        If y is None or mono-class, all samples are treated as target.
        If y is multi-class, target_class is required to select target samples.

        Preprocessing is always fitted exclusively on target samples.
        Non-target samples (if present) are stored as X_non_target_ for
        inspection; score()/score_metrics() read non-target rows straight
        from whatever (X, y) they're called with, not from this attribute.
        """
        self._validate_params()

        if y is None:
            X = check_array(X, ensure_2d=True, dtype=float)
            X_target = X
            target = self.target_class
            self.X_non_target_ = None
        else:
            X, y = check_X_y(X, y, ensure_2d=True, dtype=float)
            unique_classes = np.unique(y)
            if self.target_class is None:
                if unique_classes.size != 1:
                    raise ValueError(
                        "target_class is required when y contains more than one class"
                    )
                target = unique_classes[0]
            else:
                target = self.target_class

            X_target = X[y == target]
            if X_target.shape[0] == 0:
                raise ValueError(f"target_class {target!r} not found in y")

            X_non_target = X[y != target]
            self.X_non_target_ = X_non_target if X_non_target.shape[0] > 0 else None

        self.n_features_in_ = X.shape[1]
        self.target_class_ = target
        self.classes_ = np.array([self.negative_label, self.positive_label])
        self.n_target_samples_ = X_target.shape[0]

        # Preprocessing fitted ONLY on target samples — never on non-target
        self.preprocessor_ = self._build_preprocessor()
        if self.preprocessor_ is not None:
            self.preprocessor_.fit(X_target)
            X_target = self.preprocessor_.transform(X_target)

        self._fit_one_class_model(X_target)
        return self

    def predict(self, X):
        """Return positive_label for accepted samples, negative_label otherwise."""
        details = self.predict_details(X)
        accepted = np.asarray(details["accepted"], dtype=bool)
        return np.where(accepted, self.positive_label, self.negative_label)

    def predict_details(self, X):
        """Return per-sample diagnostics from the one-class model."""
        X = self._transform_input(X)
        details = dict(self._predict_details_one_class(X))
        details["target_class"] = self.target_class_
        return details

    def decision_function(self, X):
        """Return positive margins for accepted samples, negative for rejected."""
        X = self._transform_input(X)
        return self._predict_details_one_class(X)["decision"]

    def score(self, X, y=None):
        """Return the sklearn tuning objective.

        y=None    : acceptance rate (all samples assumed target).
        rigorous  : any single evaluation whose sensitivity on target samples
                    in X clears the nominal level (1 - self.alpha) scores
                    ~1.0, beating every evaluation that doesn't, no matter by
                    how much. Under k-fold CV, GridSearchCV averages this
                    across folds, so the mean approximates "how consistently
                    does this configuration clear the nominal sensitivity
                    across folds" — the reference-implementation criterion,
                    adapted for fold-averaging (the reference pools all
                    out-of-fold predictions into one sensitivity number
                    instead of averaging per-fold scores; see class docstring).
                    Among configurations that clear equally consistently, the
                    most complex model wins (highest self.n_components) via a
                    small tie-break term — kept tiny on purpose so a single
                    lucky fold at high n_components can't outweigh a
                    configuration that clears the bar more consistently at
                    lower n_components. Requires the subclass to expose an
                    `n_components` attribute alongside `alpha`.
        compliant : efficiency = sqrt(sensitivity * specificity).
                    Sensitivity uses target samples in X, specificity uses
                    non-target samples in X. Pair this with a splitter like
                    OCCSplit, which puts every non-target row on the test
                    side of every fold — that's what makes specificity here
                    reflect the full non-target set on every fold, not just
                    whatever fraction a plain KFold happened to hold out.
        """
        if y is None:
            y_pred = self.predict(X)
            return float(np.mean(y_pred == self.positive_label))

        # Reuse score_metrics() instead of recomputing sensitivity/specificity
        # here — same numbers, this just picks which one GridSearchCV wants
        # and swaps NaN (no valid samples) for 0.0, since sklearn can't rank NaN.
        metrics = self.score_metrics(X, y)
        sensitivity = metrics["sensitivity"]
        if not np.isfinite(sensitivity):
            return 0.0

        if self.tuning == "rigorous":
            nominal = 1.0 - self.alpha
            if sensitivity >= nominal:
                # Fixed reward for clearing the bar (not scaled by n_components) so
                # that averaging across CV folds tracks how consistently a config
                # clears, not how high n_components happened to be on a lucky fold.
                # n_components only breaks ties between equally consistent configs.
                return float(1.0 + 1e-6 * self.n_components)
            return float(sensitivity - 1.0)  # never clears the bar: best effort by sensitivity alone

        value = metrics["efficiency"]
        if not np.isfinite(value):
            value = sensitivity  # no non-target samples to score specificity against
        return float(value)

    def metrics(self, X, y) -> dict:
        """Return sensitivity, specificity, efficiency, FAR, FRR.

        Also includes specificity_by_class: specificity broken down per
        individual non-target label, in case one impostor class is harder
        to reject than another.

        Computes all metrics against whatever X and y are passed — same
        source as score_metrics(), just without the tuning-specific NaN
        handling. Use this for final evaluation on a held-out test set.
        """
        check_is_fitted(self, "target_class_")
        return one_class_metrics(
            y,
            self.predict(X),
            target_class=self.target_class_,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
        )

    def score_metrics(self, X, y) -> dict:
        """Return sensitivity, specificity, efficiency using the same convention as score().

        Sensitivity  : computed on target samples in X.
        Specificity  : computed on non-target samples in X (compliant only).
        Efficiency   : sqrt(sensitivity * specificity).

        Both figures come straight from whatever (X, y) is passed in — use
        this inside custom scorers for GridSearchCV with a splitter such as
        OCCSplit, which puts every non-target row in the test side of every
        fold, so specificity is always evaluated on the full non-target set
        regardless of which target rows that fold happens to hold out.
        """
        check_is_fitted(self, "target_class_")
        y_arr = np.asarray(y)
        is_target = y_arr == self.target_class_

        if not np.any(is_target):
            return {"sensitivity": np.nan, "specificity": np.nan, "efficiency": np.nan}

        y_pred_target = self.predict(X[is_target])
        sensitivity = float(np.mean(y_pred_target == self.positive_label))

        is_non_target = ~is_target
        if self.tuning == "rigorous" or not np.any(is_non_target):
            return {"sensitivity": sensitivity, "specificity": np.nan, "efficiency": np.nan}

        y_pred_nontarget = self.predict(X[is_non_target])
        specificity = float(np.mean(y_pred_nontarget == self.negative_label))
        efficiency = (
            float(np.sqrt(sensitivity * specificity))
            if np.isfinite(sensitivity * specificity)
            else np.nan
        )
        return {"sensitivity": sensitivity, "specificity": specificity, "efficiency": efficiency}

    # --- internal helpers ---

    def _validate_params(self):
        if self.tuning not in {"rigorous", "compliant"}:
            raise ValueError("tuning must be 'rigorous' or 'compliant'")
        if self.positive_label == self.negative_label:
            raise ValueError("positive_label and negative_label must differ")

    def _build_preprocessor(self):
        if self.preprocessor is None:
            return None
        if not hasattr(self.preprocessor, "fit") or not hasattr(self.preprocessor, "transform"):
            raise TypeError("preprocessor must implement fit and transform")
        return clone(self.preprocessor)

    def _transform_input(self, X):
        check_is_fitted(self, "target_class_")
        X = check_array(X, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if self.preprocessor_ is None:
            return X
        return self.preprocessor_.transform(X)

    # --- hooks for subclasses ---

    def _fit_one_class_model(self, X_target):
        """Fit the method-specific model on preprocessed target samples."""
        raise NotImplementedError

    def _predict_details_one_class(self, X):
        """Return a dict with at least 'accepted' (bool array) and 'decision' (float array)."""
        raise NotImplementedError


def _objective_scorer(estimator, X, y):
    return estimator.score(X, y)


def _sensitivity_scorer(estimator, X, y):
    return estimator.score_metrics(X, y)["sensitivity"]


def _specificity_scorer(estimator, X, y):
    return estimator.score_metrics(X, y)["specificity"]


def _efficiency_scorer(estimator, X, y):
    return estimator.score_metrics(X, y)["efficiency"]


# Ready-made `scoring` dict for GridSearchCV/cross_val_score on any
# OneClassMixin estimator, so score()'s tuning-specific selection criterion
# never has to be read as if it were a metric. "objective" reproduces the
# plain estimator.score() that GridSearchCV would use by default, so pass
# refit="objective" to keep the same selection behaviour while also getting
# sensitivity/specificity/efficiency as their own readable columns in
# cv_results_. specificity/efficiency come out as NaN under
# tuning="rigorous" (score_metrics() never computes them there) — harmless,
# just an uninformative column, since selection never reads them.
SCORING = {
    "objective": _objective_scorer,
    "sensitivity": _sensitivity_scorer,
    "specificity": _specificity_scorer,
    "efficiency": _efficiency_scorer,
}
