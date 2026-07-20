"""One-class modeling methods."""

from .simca import SIMCA
from .simca_plots import plot_searchcv

__all__ = ["SIMCA", "plot_searchcv"]
