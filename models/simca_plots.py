"""Plotly diagnostics for SIMCA.

Mixed into SIMCA so every plot is a direct method (``model.plot_t2q(X)``),
but plotly is only imported when a plot method actually runs — fitting or
predicting with SIMCA never requires plotly to be installed.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

_PALETTE = [
    "#6AEF6A", "#5A45FF", "#EF6565", "#d4da2f", "#ea580c",
    "#0891b2", "#38373d", "#65a30d", "#be123c", "#151ec9",
]
_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "x", "star"]
_STATUS_STYLE = {
    "accepted": ("#16a34a", "circle"),
    "rejected": ("#94a3b8", "x"),
    "extreme": ("#f97316", "diamond"),
    "outlier": ("#dc2626", "cross"),
}


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotting needs plotly. Install with `pip install plotly` "
            'or `pip install -e ".[plot]"`.'
        ) from exc
    return go


def _layout(fig, *, title, x_label, y_label, template, width, height, show_legend=True, square=False, zeroline=False):
    # showgrid=False and the box border (showline/mirror) are fixed, not
    # configurable — every plot always has a frame and never gridlines.
    axis_common = dict(showgrid=False, zeroline=zeroline, showline=True, linecolor="black", mirror=True)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template=template,
        width=width or 700,
        height=height or 550,
        xaxis=dict(title=x_label, **axis_common),
        yaxis=dict(title=y_label, **axis_common),
        showlegend=show_legend,
    )
    if square:
        # Lock the Y axis to the same scale as X: for a PCA score/loading plot,
        # distances and angles only read correctly with a 1:1 aspect ratio.
        fig.update_yaxes(scaleanchor="x", scaleratio=1)


def _group_styles(keys, style_map=None):
    """Map each unique key to a (color, symbol) pair."""
    uniques = sorted(set(keys))
    if style_map is not None:
        return {k: style_map.get(k, ("#64748b", "circle")) for k in uniques}
    return {
        k: (_PALETTE[i % len(_PALETTE)], _SYMBOLS[i % len(_SYMBOLS)])
        for i, k in enumerate(uniques)
    }


def _boundary_xy(method: str, limit: float):
    """Acceptance-boundary polyline in normalized (T2/T2_limit, Q/Q_limit) space.

    'combined', 'ci' and 'dd' all reduce to the same additive statistic
    (x + y <= limit), so they share one branch — only the limit value differs.
    """
    if method == "sim":
        return np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])
    if method == "alt":
        x = np.linspace(0.0, np.sqrt(2.0), 200)
        return x, np.sqrt(np.maximum(2.0 - x**2, 0.0))
    x = np.linspace(0.0, limit, 200)
    return x, limit - x


class SIMCAPlotMixin:
    """Plotly diagnostics for a fitted SIMCA model. Requires plotly."""

    # --- decision boundary -------------------------------------------------

    def plot_t2q(
        self,
        X,
        y=None,
        *,
        color_by: str = "status",
        log_scale: bool = False,
        title: str | None = None,
        marker_size: int = 8,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """T2 vs Q with the acceptance boundary for the fitted `method`.

        Always plots in normalized units (distance / limit) — the natural
        scale for comparing Q and T2, and the only scale that makes sense
        for method='dd', which has no raw Q/T2 limits to normalize by.

        log_scale : plot both axes in log scale — useful when a few points
        sit far beyond the boundary and squash everything else near zero.
        Points at or below 0 (can only happen in degenerate/duplicate data)
        are dropped by plotly on a log axis rather than plotted.
        """
        go = _plotly()
        check_is_fitted(self, "thresholds_")
        details = self.predict_details(X)
        method = self.method.lower()
        status = np.asarray(details["status"]).astype(str)

        yv = np.asarray(details["normalized_q"], dtype=float)
        xv = np.asarray(details["normalized_t2"], dtype=float)
        x_label = "T2 / T2 limit"

        finite_x = xv[np.isfinite(xv)]
        finite_y = yv[np.isfinite(yv)]
        extent = {"alt": np.sqrt(2.0), "sim": 1.0}.get(method, self.alpha_limit)
        x_max = max(float(finite_x.max()) if finite_x.size else 1.0, extent) * 1.15
        y_max = max(float(finite_y.max()) if finite_y.size else 1.0, extent) * 1.15

        fig = go.Figure()

        bx, by = _boundary_xy(method, self.alpha_limit)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name=f"{(1 - self.alpha) * 100:g}% limit",
        ))
        if method == "dd":
            gx, gy = _boundary_xy("combined", self.gamma_limit)
            fig.add_trace(go.Scatter(
                x=gx, y=gy, mode="lines",
                line=dict(color="#dc2626", width=2, dash="dash"),
                name=f"{(1 - self.gamma) * 100:g}% outlier limit",
            ))

        if color_by == "status":
            keys = status
            styles = _group_styles(keys, _STATUS_STYLE)
        elif color_by == "label":
            if y is None:
                raise ValueError("color_by='label' requires y")
            keys = np.asarray(y).astype(str)
            styles = _group_styles(keys)
        else:
            raise ValueError("color_by must be 'status' or 'label'")

        for key in sorted(set(keys)):
            mask = keys == key
            color, symbol = styles[key]
            fig.add_trace(go.Scatter(
                x=xv[mask], y=yv[mask], mode="markers", name=str(key),
                marker=dict(size=marker_size, color=color, symbol=symbol,
                            line=dict(width=0.7, color="black")),
                text=status[mask],
                hovertemplate=f"{x_label}=%{{x:.3f}}<br>Q ratio=%{{y:.3f}}<br>status=%{{text}}<extra></extra>",
            ))

        _layout(
            fig,
            title=title or f"SIMCA T2-Q — method={self.method!r}, alpha={self.alpha}",
            x_label=x_label, y_label="Q / Q limit",
            template=template, width=width, height=height,
        )
        if log_scale:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        else:
            fig.update_xaxes(range=[0, x_max])
            fig.update_yaxes(range=[0, y_max])
        if show:
            fig.show()
        return fig

    # --- PCA diagnostics -----------------------------------------------------

    def _pc_index(self, k: int) -> int:
        i = int(k) - 1
        if not 0 <= i < self.n_components:
            raise ValueError(f"component must be within [1, {self.n_components}], got {k}")
        return i

    def _pc_label(self, i: int) -> str:
        return f"PC{i + 1} ({self.explained_variance_ratio_[i] * 100:.1f}%)"

    def plot_scores(
        self,
        X=None,
        y=None,
        *,
        comp: tuple[int, int] = (1, 2),
        title: str | None = None,
        marker_size: int = 8,
        square: bool = True,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """Scatter of PCA scores. Uses training scores when X is None.

        square=True (default) locks both axes to the same scale, so
        distances between points on the plot reflect real distances in
        score space instead of being stretched by the figure's aspect ratio.
        """
        go = _plotly()
        check_is_fitted(self, "scores_")
        scores = self.scores_ if X is None else self.transform(X)
        i, j = self._pc_index(comp[0]), self._pc_index(comp[1])

        fig = go.Figure()
        if y is None:
            fig.add_trace(go.Scatter(
                x=scores[:, i], y=scores[:, j], mode="markers",
                marker=dict(size=marker_size, color=_PALETTE[0],
                            line=dict(width=0.7, color="black")),
                showlegend=False,
            ))
        else:
            keys = np.asarray(y).astype(str)
            styles = _group_styles(keys)
            for key in sorted(set(keys)):
                mask = keys == key
                color, symbol = styles[key]
                fig.add_trace(go.Scatter(
                    x=scores[mask, i], y=scores[mask, j], mode="markers", name=str(key),
                    marker=dict(size=marker_size, color=color, symbol=symbol,
                                line=dict(width=0.7, color="black")),
                ))

        _layout(
            fig, title=title or f"SIMCA scores ({self._pc_label(i)} vs {self._pc_label(j)})",
            x_label=self._pc_label(i), y_label=self._pc_label(j),
            template=template, width=width, height=height, square=square, zeroline=True,
        )
        if show:
            fig.show()
        return fig

    def plot_loadings(
        self,
        *,
        comp: tuple[int, int] = (1, 2),
        variable_names=None,
        mode: str = "scatter",
        title: str | None = None,
        square: bool = True,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """Loadings as a scatter (comp i vs comp j) or as line profiles per variable.

        square (scatter mode only) locks both axes to the same scale, same
        reasoning as plot_scores — it has no meaning in line mode, where the
        X axis is variable index, not a loading, so it's ignored there.
        """
        go = _plotly()
        check_is_fitted(self, "loadings_")
        names = list(variable_names) if variable_names is not None else [
            f"var{k + 1}" for k in range(self.loadings_.shape[0])
        ]

        fig = go.Figure()
        if mode == "scatter":
            i, j = self._pc_index(comp[0]), self._pc_index(comp[1])
            fig.add_trace(go.Scatter(
                x=self.loadings_[:, i], y=self.loadings_[:, j], mode="markers+text",
                text=names, textposition="top center",
                marker=dict(size=7, color=_PALETTE[0]),
            ))
            x_label, y_label = self._pc_label(i), self._pc_label(j)
        elif mode == "line":
            x = np.arange(len(names))
            for k in range(self.n_components):
                fig.add_trace(go.Scatter(
                    x=x, y=self.loadings_[:, k], mode="lines",
                    name=self._pc_label(k), line=dict(color=_PALETTE[k % len(_PALETTE)]),
                ))
            fig.update_xaxes(tickvals=x, ticktext=names)
            x_label, y_label = "variable", "loading"
        else:
            raise ValueError("mode must be 'scatter' or 'line'")

        _layout(
            fig, title=title or "SIMCA loadings", x_label=x_label, y_label=y_label,
            template=template, width=width, height=height, show_legend=(mode == "line"),
            square=(square and mode == "scatter"), zeroline=(mode == "scatter"),
        )
        if show:
            fig.show()
        return fig

    def plot_scores_loadings(
        self,
        X=None,
        y=None,
        *,
        comp: tuple[int, int] = (1, 2),
        loadings_mode: str = "scatter",
        variable_names=None,
        title: str | None = None,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """Scores and loadings side by side, as two separate subplots (not a biplot)."""
        from plotly.subplots import make_subplots
        _plotly()

        i, j = self._pc_index(comp[0]), self._pc_index(comp[1])
        scores_fig = self.plot_scores(X, y, comp=comp, template=template)
        loadings_fig = self.plot_loadings(
            mode=loadings_mode, comp=comp, variable_names=variable_names, template=template,
        )

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Scores", "Loadings"), horizontal_spacing=0.12)
        for tr in scores_fig.data:
            fig.add_trace(tr, row=1, col=1)
        for tr in loadings_fig.data:
            fig.add_trace(tr, row=1, col=2)

        fig.update_xaxes(title_text=self._pc_label(i), row=1, col=1)
        fig.update_yaxes(title_text=self._pc_label(j), row=1, col=1, scaleanchor="x", scaleratio=1)
        if loadings_mode == "scatter":
            fig.update_xaxes(title_text=self._pc_label(i), row=1, col=2)
            fig.update_yaxes(title_text=self._pc_label(j), row=1, col=2, scaleanchor="x2", scaleratio=1)
        else:
            fig.update_xaxes(title_text="variable", row=1, col=2)
            fig.update_yaxes(title_text="loading", row=1, col=2)

        fig.update_layout(
            title=dict(text=title or "Scores & Loadings", x=0.5),
            template=template, width=width or 1100, height=height or 520,
            showlegend=True,
        )
        # Same fixed frame/no-gridlines rule as every other plot, applied by
        # hand here since make_subplots axes don't go through _layout().
        for axis_name in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
            fig.layout[axis_name].update(showgrid=False, showline=True, linecolor="black", mirror=True)
        fig.layout.xaxis.zeroline = True
        fig.layout.yaxis.zeroline = True
        fig.layout.xaxis2.zeroline = (loadings_mode == "scatter")
        fig.layout.yaxis2.zeroline = (loadings_mode == "scatter")
        if show:
            fig.show()
        return fig

    def plot_scree(
        self,
        *,
        title: str | None = None,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """Explained and cumulative variance (%) per retained component, as scatter+lines."""
        go = _plotly()
        check_is_fitted(self, "explained_variance_ratio_")
        ratio = np.asarray(self.explained_variance_ratio_, dtype=float) * 100.0
        cumulative = np.cumsum(ratio)
        x = np.arange(1, len(ratio) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ratio, mode="lines+markers",
                                  name="explained", line=dict(color=_PALETTE[0]),
                                  marker=dict(size=8, line=dict(width=0.7, color="black"))))
        fig.add_trace(go.Scatter(x=x, y=cumulative, mode="lines+markers",
                                  name="cumulative", line=dict(color=_PALETTE[1]),
                                  marker=dict(size=8, line=dict(width=0.7, color="black"))))
        _layout(
            fig, title=title or "SIMCA scree plot", x_label="component", y_label="variance (%)",
            template=template, width=width, height=height,
        )
        fig.update_xaxes(tickvals=x)
        if show:
            fig.show()
        return fig

    def plot_contributions(
        self,
        X,
        *,
        kind: str = "q",
        variable_names=None,
        title: str | None = None,
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """Per-variable contribution to Q ('q') or T2 ('t2').

        Contributions are sign(raw) * raw**2, so for a single sample the
        bars sum (in absolute value) to that sample's Q or T2 exactly.
        A single sample in X gives signed contributions for that sample;
        multiple samples give the mean absolute contribution across them.
        """
        go = _plotly()
        check_is_fitted(self, "thresholds_")
        if kind == "q":
            contrib = self.residual_contributions(X)
        elif kind == "t2":
            contrib = self.score_contributions(X)
        else:
            raise ValueError("kind must be 'q' or 't2'")

        names = list(variable_names) if variable_names is not None else [
            f"var{k + 1}" for k in range(contrib.shape[1])
        ]
        signed = contrib.shape[0] == 1
        values = contrib[0] if signed else np.mean(np.abs(contrib), axis=0)

        # One fixed color regardless of sign — sign is already visible from
        # the bar direction itself (signed case), no need to double-encode it.
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=_PALETTE[0]))
        ylabel = f"{kind.upper()} contribution" + ("" if signed else " (mean |.|)")
        _layout(
            fig, title=title or f"SIMCA {kind.upper()} contributions", x_label="variable", y_label=ylabel,
            template=template, width=width, height=height, show_legend=False,
        )
        if show:
            fig.show()
        return fig

    # --- classification diagnostics ------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        *,
        title: str | None = None,
        show_percent: bool = False,
        colorscale: str = "Blues",
        template: str = "plotly_white",
        show: bool = False,
        width: int | None = None,
        height: int | None = None,
    ):
        """One-class confusion matrix: one column per class, accepted/rejected rows."""
        go = _plotly()
        check_is_fitted(self, "target_class_")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")

        classes = sorted(np.unique(y_true).tolist())
        if self.target_class_ in classes:
            classes.remove(self.target_class_)
            classes = [self.target_class_] + classes

        accepted = np.array([
            np.sum((y_true == cls) & (y_pred == self.positive_label)) for cls in classes
        ])
        rejected = np.array([
            np.sum((y_true == cls) & (y_pred == self.negative_label)) for cls in classes
        ])
        totals = accepted + rejected
        z = np.vstack([rejected, accepted])  # row0 = rejected (top), row1 = accepted (bottom)

        if show_percent:
            pct = np.where(totals > 0, z / totals * 100, 0.0)
            text = [[f"{z[r, c]}<br>({pct[r, c]:.1f}%)" for c in range(len(classes))] for r in range(2)]
        else:
            text = [[str(v) for v in row] for row in z]

        labels = [f"<b>{c}</b>" if c == self.target_class_ else str(c) for c in classes]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=["rejected", "accepted"], colorscale=colorscale,
            showscale=False, text=text, texttemplate="%{text}",
            hovertemplate="%{x}<br>%{y}: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title or f"SIMCA confusion matrix (target={self.target_class_!r})", x=0.5),
            template=template, width=width or 550, height=height or 420,
            xaxis=dict(title="class"), yaxis=dict(title="prediction", autorange="reversed"),
        )
        if show:
            fig.show()
        return fig


def plot_searchcv(
    grid_search,
    *,
    x_param: str | None = None,
    group_param: str | None = None,
    y_metric: str | None = None,
    title: str | None = None,
    score_label: str | None = None,
    template: str = "plotly_white",
    show: bool = False,
    width: int | None = None,
    height: int | None = None,
):
    """Plot GridSearchCV mean +/- std test score against one hyperparameter.

    Not a SIMCA method (it plots a fitted GridSearchCV, not a fitted model),
    so it stays a standalone function.

    y_metric selects which column of cv_results_ to plot when GridSearchCV
    was built with a `scoring` dict of several named scorers (looks for
    mean_test_<y_metric> / std_test_<y_metric>). Without it, this plots
    whatever the single selection score is (mean_test_score) — for SIMCA's
    tuning="rigorous" that is NOT sensitivity, it's an internal selection
    criterion. Pass a multi-metric `scoring` dict to GridSearchCV (e.g. one
    scorer computing score_metrics()["sensitivity"]) and y_metric="sensitivity"
    here to plot the real, interpretable value instead.
    """
    go = _plotly()
    if not hasattr(grid_search, "cv_results_"):
        raise ValueError("grid_search must be a fitted GridSearchCV instance")

    cv = grid_search.cv_results_
    param_names = [k[len("param_"):] for k in cv if k.startswith("param_")]
    if not param_names:
        raise ValueError("cv_results_ has no 'param_*' columns")

    if x_param is None:
        x_param = max(param_names, key=lambda n: len(set(cv[f"param_{n}"])))
    remaining = [n for n in param_names if n != x_param]
    if group_param is None:
        group_param = remaining[0] if remaining else None

    mean_key = f"mean_test_{y_metric}" if y_metric else "mean_test_score"
    std_key = f"std_test_{y_metric}" if y_metric else "std_test_score"
    if mean_key not in cv:
        available = sorted(k[len("mean_test_"):] for k in cv if k.startswith("mean_test_"))
        raise ValueError(
            f"{mean_key!r} not in cv_results_. Available metrics: {available}. "
            "Pass one of them as y_metric (matches the scorer names given to "
            "GridSearchCV(scoring=...))."
        )
    mean = np.asarray(cv[mean_key], dtype=float)
    std = np.asarray(cv[std_key], dtype=float)
    x_vals = np.asarray(cv[f"param_{x_param}"])
    groups = np.asarray(cv[f"param_{group_param}"]).astype(str) if group_param else np.zeros(len(mean), dtype="<U1")

    ylabel = score_label or y_metric or "mean CV score"

    fig = go.Figure()
    for k, grp in enumerate(sorted(set(groups))):
        mask = groups == grp
        order = np.argsort(x_vals[mask])
        gx, gy, gstd = x_vals[mask][order], mean[mask][order], std[mask][order]
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines+markers",
            error_y=dict(type="data", array=gstd, visible=True),
            line=dict(color=_PALETTE[k % len(_PALETTE)]),
            name=str(grp) if group_param else ylabel,
        ))

    best = getattr(grid_search, "best_params_", {}) or {}
    if x_param in best:
        fig.add_vline(x=best[x_param], line_dash="dash", line_color="red",
                       annotation_text=f"best {x_param}={best[x_param]}")

    _layout(
        fig,
        title=title or f"GridSearchCV — {ylabel} vs {x_param}",
        x_label=x_param, y_label=ylabel, template=template, width=width, height=height,
        show_legend=bool(group_param),
    )
    if show:
        fig.show()
    return fig
