"""Benchmark comparison stubs for future integration.

Placeholder structure for comparing model results against operational
forecasting systems (GloFAS) and observed discharge databases (GRDC).
"""

from __future__ import annotations


class BenchmarkComparison:
    """Compare model results against external benchmarks."""

    @staticmethod
    def compare_with_glofas(model_results: dict, glofas_data) -> dict:
        """Compare model predictions with GloFAS reforecasts.

        Parameters
        ----------
        model_results : dict
            Per-basin model results from :class:`EvaluationPipeline`.
        glofas_data
            GloFAS reforecast data (format TBD).

        Returns
        -------
        dict
            Comparison metrics (stub -- returns empty dict).
        """
        # TODO: implement GloFAS comparison once data format is finalised
        return {}

    @staticmethod
    def compare_with_grdc(model_results: dict, grdc_data) -> dict:
        """Compare model predictions with GRDC observations.

        Parameters
        ----------
        model_results : dict
            Per-basin model results from :class:`EvaluationPipeline`.
        grdc_data
            GRDC observation data (format TBD).

        Returns
        -------
        dict
            Comparison metrics (stub -- returns empty dict).
        """
        # TODO: implement GRDC comparison once data format is finalised
        return {}
