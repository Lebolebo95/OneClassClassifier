"""Metrics for one-class class modeling."""

from __future__ import annotations

import numpy as np


def one_class_metrics(
    y_true,
    y_pred,
    *,
    target_class=None,
    positive_label=1,
    negative_label=-1,
) -> dict:
    """Return sensitivity, specificity, efficiency, FAR, FRR.

    y_pred contains acceptance labels: positive_label = accepted, negative_label = rejected.
    Samples in y_true equal to target_class are target; all others are non-target.

    specificity is pooled across every non-target sample; specificity_by_class
    breaks it down per individual non-target label, e.g. to see whether one
    impostor class is harder to reject than another.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    is_target = y_true == target_class if target_class is not None else y_true == positive_label
    accepted = y_pred == positive_label

    n_target = int(np.sum(is_target))
    n_non_target = int(np.sum(~is_target))

    sensitivity = float(np.mean(accepted[is_target])) if n_target else np.nan # True Positive Rate
    specificity = float(np.mean(~accepted[~is_target])) if n_non_target else np.nan # True Negative Rate
    efficiency = float(np.sqrt(sensitivity * specificity)) if np.isfinite(sensitivity * specificity) else np.nan # Geometric mean of sensitivity and specificity
    far = float(1.0 - specificity) if np.isfinite(specificity) else np.nan # False Acceptance Rate
    frr = float(1.0 - sensitivity) if np.isfinite(sensitivity) else np.nan # False Rejection Rate

    specificity_by_class = {
        label: float(np.mean(~accepted[y_true == label]))
        for label in np.unique(y_true[~is_target])
    }

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "specificity_by_class": specificity_by_class,
        "efficiency": efficiency,
        "FAR": far,
        "FRR": frr,
        "n_target": n_target,
        "n_non_target": n_non_target,
    }
