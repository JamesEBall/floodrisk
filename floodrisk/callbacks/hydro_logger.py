"""Callback that logs hydrograph plots and per-basin summaries."""

from __future__ import annotations

import logging
from pathlib import Path

from floodrisk.torchharness import Callback

logger = logging.getLogger(__name__)


class HydroLogger(Callback):
    """Log hydrograph plots and per-basin NSE summaries during validation.

    Parameters
    ----------
    output_dir:
        Root directory for saving artefacts.
    sample_basins:
        Explicit list of basin IDs to plot.  When *None*, the first
        ``n_samples`` basins encountered during validation are used.
    n_samples:
        Number of basins to plot when ``sample_basins`` is not given.
    """

    def __init__(
        self,
        output_dir: str,
        sample_basins: list[str] | None = None,
        n_samples: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sample_basins = sample_basins
        self.n_samples = n_samples

    def on_validation_epoch_end(
        self, epoch: int, metrics: dict, trainer: object
    ) -> None:
        epoch_dir = self.output_dir / "hydrographs" / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Attempt to retrieve per-basin results stashed by the trainer or
        # a preceding callback.  The trainer is not required to populate
        # this, so we guard against its absence.
        basin_results: dict | None = metrics.get("basin_results")
        if basin_results is None:
            logger.debug(
                "No per-basin results available; skipping hydrograph plots."
            )
            return

        basins_to_plot = self.sample_basins or list(basin_results.keys())[
            : self.n_samples
        ]

        self._plot_hydrographs(basins_to_plot, basin_results, epoch_dir)
        self._log_basin_summary(basin_results, epoch)

    def _plot_hydrographs(
        self,
        basins: list[str],
        basin_results: dict,
        epoch_dir: Path,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib not installed; skipping hydrograph plots."
            )
            return

        for basin_id in basins:
            result = basin_results.get(basin_id)
            if result is None:
                continue
            obs = result["observed"]
            pred = result["predicted"]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(obs, label="Observed", linewidth=1.0)
            ax.plot(pred, label="Predicted", linewidth=1.0, alpha=0.8)
            ax.set_title(f"Basin {basin_id}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Streamflow")
            ax.legend()
            fig.tight_layout()
            fig.savefig(epoch_dir / f"{basin_id}.png", dpi=100)
            plt.close(fig)

        logger.info("Saved hydrograph plots to %s", epoch_dir)

    @staticmethod
    def _log_basin_summary(basin_results: dict, epoch: int) -> None:
        nse_scores: dict[str, float] = {}
        for basin_id, result in basin_results.items():
            nse = result.get("nse")
            if nse is not None:
                nse_scores[basin_id] = nse

        if not nse_scores:
            return

        sorted_basins = sorted(nse_scores.items(), key=lambda x: x[1])
        n_report = min(5, len(sorted_basins))
        worst = sorted_basins[:n_report]
        best = sorted_basins[-n_report:]

        logger.info("Epoch %d basin NSE summary:", epoch)
        logger.info(
            "  Best:  %s",
            ", ".join(f"{b}={v:.4f}" for b, v in reversed(best)),
        )
        logger.info(
            "  Worst: %s",
            ", ".join(f"{b}={v:.4f}" for b, v in worst),
        )
