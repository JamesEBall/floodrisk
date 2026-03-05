"""Evaluation visualisation utilities."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_hydrograph(observed, predicted, basin_id: str, save_path=None):
    """Line plot of observed vs predicted streamflow.

    Parameters
    ----------
    observed : array-like
        Observed streamflow values.
    predicted : array-like
        Predicted streamflow values.
    basin_id : str
        Basin identifier (used in title).
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(observed, label="Observed", linewidth=0.8)
    ax.plot(predicted, label="Predicted", linewidth=0.8, alpha=0.8)
    ax.set_title(f"Hydrograph -- Basin {basin_id}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Streamflow")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_nse_map(basin_nse: dict, save_path=None):
    """Scatter plot of basin NSE values on a lat/lon plane.

    Parameters
    ----------
    basin_nse : dict
        Mapping of ``basin_id`` to ``{"nse": float, "lat": float, "lon": float}``.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lats = [v["lat"] for v in basin_nse.values()]
    lons = [v["lon"] for v in basin_nse.values()]
    nse_vals = [v["nse"] for v in basin_nse.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(lons, lats, c=nse_vals, cmap="RdYlGn", vmin=0, vmax=1, s=20)
    fig.colorbar(sc, ax=ax, label="NSE")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Basin-level NSE")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scatter(observed, predicted, save_path=None):
    """Scatter plot of observed vs predicted with a 1:1 reference line.

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Predicted values.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(observed, predicted, alpha=0.3, s=5)
    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="1:1")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Observed vs Predicted")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_nse_cdf(nse_values, save_path=None):
    """CDF curve of basin NSE values.

    Parameters
    ----------
    nse_values : array-like
        Collection of per-basin NSE values.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    nse_values = np.sort(np.asarray(nse_values))
    cdf = np.arange(1, len(nse_values) + 1) / len(nse_values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nse_values, cdf, linewidth=1.5)
    ax.set_xlabel("NSE")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Basin NSE")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
