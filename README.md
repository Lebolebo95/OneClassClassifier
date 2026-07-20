# OneClassClassifier

A **SIMCA** one-class classifier, compatible with the scikit-learn API (`fit`/`predict`/`score`/`GridSearchCV`).

Fits a PCA model on samples from a single target class, then classifies new samples by how far they deviate from that model, using the **Q** (residual outside the PCA subspace) and **T2** (Hotelling distance inside the subspace) statistics. No labeled examples of other classes are needed to build the model — they're only used, if available, to measure how well it rejects them.

See `wine_demo.ipynb` for a complete worked example (Wine dataset, cross-validated tuning, held-out test evaluation, all diagnostic plots).

## Structure

```
OneClassClassifier/
├── pyproject.toml     # package metadata + dependencies, for "pip install -e ."
├── __init__.py        # exports OneClassMixin, SIMCA, plot_searchcv, one_class_metrics, OCCSplit
├── base.py             # OneClassMixin: fit/predict/score shared by every one-class model
├── metrics.py          # one_class_metrics: sensitivity, specificity, efficiency, FAR, FRR
├── splitOCC.py         # OCCSplit: cross-validation splitter for one-class models
├── wine_demo.ipynb     # worked example on the Wine dataset
└── models/
    ├── __init__.py      # exports SIMCA, plot_searchcv
    ├── simca.py          # the SIMCA class
    ├── simca_limits.py   # statistical limits (Q, T2, DD-SIMCA) used by simca.py
    └── simca_plots.py    # plotly diagnostics, mixed into SIMCA (optional, see below)
```

## Installation

Editable install (recommended — `from OneClassClassifier import SIMCA` then works from any folder/notebook, no `sys.path` needed):

```bash
pip install -e /path/to/OneClassClassifier
```

Dependencies: `numpy`, `scipy`, `scikit-learn`. Plotting is an optional extra (see below).

`wine_demo.ipynb` works even without an editable install: its first cell tries the normal import first, and only falls back to inserting the repo's parent directory into `sys.path` if that fails — so opening it straight after cloning, before running `pip install`, still works.

## Quickstart

```python
from OneClassClassifier import SIMCA

# X_train contains only target-class samples
model = SIMCA(n_components=2, method="alt", alpha=0.05).fit(X_train)

y_pred = model.predict(X_test)           # 1 = accepted, -1 = rejected
details = model.predict_details(X_test)  # q, t2, statistic, limits, status, ...
```

For a multi-class dataset, pass `y` and say which class to model:

```python
model = SIMCA(n_components=2, target_class="olive_oil").fit(X, y)
metrics = model.metrics(X_test, y_test)  # sensitivity, specificity, efficiency, FAR, FRR
```

## API

### `SIMCA(...)`

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | `2` | number of principal components |
| `method` | `"alt"` | decision rule (see below) |
| `alpha` | `0.05` | significance level of the acceptance limit |
| `gamma` | `0.01` | significance level of the outlier limit (`method="dd"` only) |
| `q_threshold` | `"jm"` | Q-limit estimation method |
| `t2_threshold` | `"fdist"` | T2-limit estimation method |
| `center` | `True` | mean-center the data before PCA |
| `eps` | `1e-12` | regularization constant |
| `target_class` | `None` | class to model (required when `y` has more than one class) |
| `preprocessor` | `None` | transformer fitted only on target-class samples |
| `tuning` | `"rigorous"` | `score()` strategy for `GridSearchCV` (see below) |
| `positive_label` / `negative_label` | `1` / `-1` | labels for accepted/rejected samples |

### Decision rules (`method`)

| `method` | Acceptance rule |
|---|---|
| `sim` | `Q <= q_limit` **and** `T2 <= t2_limit` (rectangular boundary) |
| `alt` | `sqrt((Q/q_limit)^2 + (T2/t2_limit)^2) <= sqrt(2)` (elliptical boundary) |
| `combined` / `ci` | chi-square combined statistic of Q and T2 |
| `dd` | DD-SIMCA: chi-square with moment-matched parameters on Q and T2, plus a stricter limit (`gamma`) separating "extreme" class members from true outliers |

### Q/T2 limit methods (`q_threshold`, `t2_threshold`)

`jm` (Jackson-Mudholkar, Q only), `chi2box`, `chi2`, `chi2pom` (degrees of freedom rounded to an integer), `percentile` (empirical, from the training data); for T2 also `fdist`/`fdistrig` (F-distribution, standard/rigorous).

### `tuning`: `rigorous` vs `compliant`

Controls what `score()` returns (used by `GridSearchCV`):

- `rigorous`: rewards the most complex model (highest `n_components`) among configurations whose sensitivity on the target samples in `X` clears the nominal level `1 - alpha`. A configuration that clears the nominal level always beats one that doesn't, no matter by how much. If none clear it, the configuration with the highest sensitivity wins instead.
- `compliant`: efficiency = `sqrt(sensitivity * specificity)`, both computed straight from whatever non-target samples happen to be in `X` — pair this with `OCCSplit` (below), which guarantees that's always the complete non-target set, on every fold.

Neither of these is meant to be read as a plain metric — use `score_metrics()` or `metrics()` for that. See `wine_demo.ipynb` for a pattern that reports real sensitivity alongside the tuning score.

### Cross-validation: `OCCSplit`

A plain `KFold`/`StratifiedKFold` splits every row, including non-target ones — but SIMCA never fits on non-target rows, so they have no business being "held out" from training the way genuine training data would be. `OCCSplit` splits only the target-class rows into folds; every non-target row is placed on the *test* side of every fold, alongside whichever target rows that fold holds out. `train_idx` therefore never contains non-target rows at all. It's a drop-in `cv=` replacement anywhere scikit-learn accepts one (`GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `cross_validate`).

```python
from sklearn.model_selection import GridSearchCV, KFold
from OneClassClassifier import SIMCA, OCCSplit

cv = OCCSplit(target_class=1, cv=KFold(5, shuffle=True, random_state=0))
GridSearchCV(SIMCA(target_class=1), param_grid, cv=cv).fit(X, y)
```

`cv` accepts any scikit-learn splitter (`KFold`, `RepeatedKFold`, `LeaveOneOut`, ...) to control how the target-class rows themselves are divided; it defaults to `KFold(n_splits=5)`.

### Common methods (from `OneClassMixin`)

- `fit(X, y=None)` — fits only on target samples; the preprocessor (if any) is fitted exclusively on them. Non-target rows (if any) are stored as `X_non_target_` for inspection only — nothing in `score()`/`score_metrics()` reads it.
- `predict(X)` — `positive_label`/`negative_label` per sample.
- `predict_details(X)` — dict with `q`, `t2`, `statistic`, limits, `status` (`accepted`/`rejected`, or `accepted`/`extreme`/`outlier` for `dd`).
- `decision_function(X)` — signed margin (positive = accepted).
- `score(X, y=None)` — the `GridSearchCV` objective, per `tuning`.
- `score_metrics(X, y)` — `{sensitivity, specificity, efficiency}` dict, same convention as `score()`, computed straight from the `X, y` passed in.
- `metrics(X, y)` — `sensitivity`, `specificity`, `specificity_by_class` (broken down per non-target label), `efficiency`, `FAR`, `FRR` computed on the given set (for final evaluation on a held-out test set).

### SIMCA-specific methods

- `transform(X)` / `inverse_transform(scores)` — project onto the PCA subspace and reconstruct.
- `residual_contributions(X)` — signed per-variable contribution to Q (`sign(r) * r**2`, so contributions sum to Q in absolute value).
- `score_contributions(X)` — signed per-variable contribution to T2, same convention.
- `q_limit`, `t2_limit`, `alpha_limit`, `gamma_limit` — limits estimated at fit time.

## Plots (plotly, optional)

`plotly` is not a base dependency: `fit`/`predict`/`score` work without it. It's only needed for the `plot_*` methods, imported on first use — if it's missing, the error message says so clearly. To install it: `pip install -e ".[plot]"` (or `pip install plotly`).

```python
model = SIMCA(n_components=2, method="dd", target_class="A").fit(X, y)
model.plot_t2q(X, y, log_scale=False).show()   # acceptance boundary + points colored by status
model.plot_scores(X, y).show()                 # PCA score scatter (1:1 aspect ratio)
model.plot_loadings(mode="line").show()
model.plot_loadings(mode="scatter").show()     # 1:1 aspect ratio
model.plot_scores_loadings(X, y).show()        # scores and loadings, side by side
model.plot_scree().show()
model.plot_contributions(X[:1], kind="q").show()   # per-variable contributions for one sample
model.plot_confusion_matrix(y_test, model.predict(X_test)).show()
```

`plot_searchcv(grid_search)` is a standalone function (not a method) because it plots a fitted `GridSearchCV`, not a `SIMCA`. Pass `y_metric` to plot a real metric instead of the raw tuning score (see the `scoring` pattern in `wine_demo.ipynb`):

```python
from OneClassClassifier import plot_searchcv
plot_searchcv(gs, x_param="n_components", group_param="method", y_metric="sensitivity").show()
```

## Status

No automated test suite yet. Correctness has been checked manually: numeric comparisons against prior versions of the code across randomized configurations, and cross-checks against a reference implementation of SIMCA/DD-SIMCA for the statistical formulas and the cross-validation scheme.
