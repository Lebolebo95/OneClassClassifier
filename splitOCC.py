"""Cross-validation splitter for one-class models."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold


class OCCSplit(BaseCrossValidator):
    """Cross-validation splitter for one-class models.

    Splits only the target-class rows into folds (using `cv`, any
    sklearn-compatible splitter); every non-target row is placed on the
    *test* side of every fold, alongside whichever target rows that fold
    holds out. train_idx therefore contains only target rows — non-target
    rows never reach fit() at all, matching what they actually are: pure
    evaluation data that never influences the model, so there's nothing to
    protect by holding some of them back from evaluation the way you would
    with genuine training data.

    Pair this with score_metrics()/score() (see OneClassMixin), which read
    non-target rows straight from whatever is passed to them at scoring
    time — since every fold's test side always carries the complete
    non-target set, specificity is evaluated against all of it on every
    fold, not just whatever a plain KFold happened to leave out.

    Drop-in replacement for KFold/StratifiedKFold wherever sklearn accepts
    a `cv` argument: GridSearchCV, RandomizedSearchCV, cross_val_score,
    cross_validate, ...

    Parameters
    ----------
    target_class : label
        The class to split into folds. Every other label in y is treated
        as non-target and kept out of every test fold.
    cv : cross-validation splitter, default=KFold(n_splits=5)
        Any sklearn splitter (KFold, RepeatedKFold, LeaveOneOut, ...) used
        to divide the target-class rows. StratifiedKFold works too but adds
        nothing here, since the rows it sees all share the same label.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV, KFold
    >>> cv = OCCSplit(target_class=1, cv=KFold(5, shuffle=True, random_state=0))
    >>> GridSearchCV(SIMCA(target_class=1), param_grid, cv=cv).fit(X, y)
    """

    def __init__(self, target_class, cv=None):
        self.target_class = target_class
        self.cv = cv if cv is not None else KFold(n_splits=5)

    def _target_indices(self, y):
        return np.flatnonzero(np.asarray(y) == self.target_class)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        if y is None:
            raise ValueError("OCCSplit requires y to identify the target class")
        y = np.asarray(y)
        target_idx = self._target_indices(y)
        non_target_idx = np.flatnonzero(y != self.target_class)
        for _, test_pos in self.cv.split(target_idx, y[target_idx]):
            # BaseCrossValidator.split() builds train_idx as "everything not
            # in this test set" — folding non_target_idx into every yielded
            # test set is what keeps it out of train_idx on every fold.
            yield np.concatenate([target_idx[test_pos], non_target_idx])

    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None:
            return self.cv.get_n_splits()
        y = np.asarray(y)
        target_idx = self._target_indices(y)
        return self.cv.get_n_splits(target_idx, y[target_idx])
