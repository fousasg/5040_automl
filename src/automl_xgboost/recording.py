"""Per-run trial recording across all optimization studies.

The pipeline creates several Optuna studies (one Round 1 study per iteration,
plus a Round 2 feature-selection study). This module merges their trials into
the rows that land in ``trials.csv`` and draws the combined history plots, so
every round/iteration is visible and raw scores sit next to penalized ones.
"""

from __future__ import annotations

import shutil
from typing import Any

import pandas as pd

from .config import AutoMLConfig
from .exceptions import ArtifactError
from .metrics import Direction, running_best


class TrialRecorder:
    """Collects one row per trial from every study in a run."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.iterative = config.iterations > 1 or config.feature_selection
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        trial: Any,
        *,
        round_name: str,
        iteration: int,
        raw_score: float | None,
        penalized_score: float | None,
        selected_features: list[str] | None,
    ) -> None:
        unique = iteration * 10_000 + trial.number if self.iterative else trial.number
        self.rows.append(
            {
                "trial_number": unique,
                "round": round_name,
                "iteration": iteration,
                "trial_number_within_round": trial.number,
                "state": "COMPLETE" if raw_score is not None else "FAIL",
                "objective": penalized_score,
                "raw_score": raw_score,
                "penalized_score": penalized_score,
                "algorithm": "xgboost",
                "selected_features": ",".join(selected_features or []),
            }
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def merge_trials_dataframe(
    studies: list[tuple[str, int, Any]], *, iterative: bool
) -> pd.DataFrame:
    """Fallback rebuild directly from the Optuna studies when no recorder exists."""

    rows: list[dict[str, Any]] = []
    for round_name, iteration, study in studies:
        for trial in study.trials:
            unique = iteration * 10_000 + trial.number if iterative else trial.number
            raw = trial.user_attrs.get("cv_score")
            if raw is None:
                raw = trial.value if trial.user_attrs.get("selected_features") is not None else None
            rows.append(
                {
                    "trial_number": unique,
                    "round": round_name,
                    "iteration": iteration,
                    "trial_number_within_round": trial.number,
                    "state": trial.state.name,
                    "objective": trial.value,
                    "raw_score": raw,
                    "penalized_score": trial.value,
                    "algorithm": "xgboost",
                    "selected_features": ",".join(trial.user_attrs.get("selected_features") or []),
                    "started_at": trial.datetime_start,
                    "completed_at": trial.datetime_complete,
                    "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
                    "failure_reason": trial.user_attrs.get("failure_reason"),
                    **{
                        f"param_{key}": value
                        for key, value in (trial.user_attrs.get("normalized_parameters") or {}).items()
                    },
                }
            )
    return pd.DataFrame(rows)


def write_trials_artifacts(
    trials_frame: pd.DataFrame,
    scorer: str,
    direction: Direction,
    run_dir: Any,
) -> tuple[Any, Any, Any]:
    """Write trials.csv plus the Plotly HTML and matplotlib PNG histories.

    The x-axis shows sequential trial number within each round (1-based).
    A secondary x-axis at the top marks the iteration number so both
    dimensions are visible.
    """

    import os
    from pathlib import Path

    run_dir = Path(run_dir)
    trials_path = run_dir / "trials.csv"
    trials_frame.to_csv(trials_path, index=False)

    complete = trials_frame[
        (trials_frame["state"] == "COMPLETE") & trials_frame["penalized_score"].notna()
    ]
    if complete.empty:
        history = run_dir / "optimization_history.html"
        history.write_text("<html><body><p>No completed trials.</p></body></html>")
        progress = run_dir / "optimization_progress.png"
        progress.write_bytes(b"")
        return trials_path, history, progress

    complete = complete.sort_values(["iteration", "round", "trial_number_within_round"]).copy()
    # X-axis: sequential trial index across the full run (1-based)
    complete["x"] = range(1, len(complete) + 1)
    total_iterations = int(complete["iteration"].max())
    rounds = list(complete["round"].unique())

    def _tick_positions() -> tuple[list[int], list[str]]:
        """Return (x positions, iteration labels) at the END of each round1 block."""
        positions: list[int] = []
        labels: list[str] = []
        for iteration in range(1, total_iterations + 1):
            subset = complete[
                (complete["iteration"] == iteration) & (complete["round"] == "round1")
            ] if "round1" in rounds else complete[complete["iteration"] == iteration]
            if not subset.empty:
                positions.append(int(subset["x"].iloc[-1]))
                labels.append(f"iter {iteration}")
        return positions, labels

    def _round_boundaries() -> list[tuple[int, str]]:
        """Return (midpoint, round name) for each distinct round block."""
        boundaries: list[tuple[int, str]] = []
        for name in rounds:
            subset = complete[complete["round"] == name]
            if not subset.empty:
                boundaries.append((subset["x"].mean(), name))
        return boundaries

    def _iteration_axis(ax: Any) -> None:
        secondary = ax.secondary_xaxis("top")
        positions, labels = _tick_positions()
        if positions:
            secondary.set_xticks(positions)
            secondary.set_xticklabels(labels, fontsize=8, rotation=30, ha="left")
            secondary.set_xlabel("Iteration")
        # Light vertical separators at iteration boundaries
        for pos, _ in zip(positions, labels):
            ax.axvline(pos, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for name in rounds:
            subset = complete[complete["round"] == name]
            fig.add_scatter(
                x=subset["x"].tolist(),
                y=subset["penalized_score"].tolist(),
                mode="markers",
                name=f"{name} score",
                customdata=subset[["iteration", "trial_number_within_round", "raw_score"]].values.tolist(),
                hovertemplate="Iteration %{customdata[0]} Trial %{customdata[1]}<br>Raw %{customdata[2]:.4f}<extra></extra>",
            )
        incumbents = running_best(
            [float(v) for v in complete["penalized_score"].tolist()], direction
        )
        fig.add_scatter(
            x=complete["x"].tolist(), y=incumbents, mode="lines+markers",
            name="running best", line={"color": "black", "width": 2},
        )
        tick_pos, tick_labels = _tick_positions()
        if tick_pos:
            ymax = max(complete["penalized_score"].tolist())
            fig.add_scatter(
                x=tick_pos,
                y=[ymax * 1.02 for _ in tick_pos],
                mode="text",
                text=tick_labels,
                textposition="top center",
                showlegend=False,
                textfont={"size": 9, "color": "gray"},
            )
        # Vertical dashed lines at round1->round2 boundaries
        boundaries = _round_boundaries()
        for midpoint, name in boundaries:
            fig.add_vline(x=midpoint, line_dash="dot", line_color="gray", opacity=0.3)
        # Round annotations
        for midpoint, name in boundaries:
            fig.add_annotation(
                x=midpoint, y=complete["penalized_score"].max() * 1.01,
                text=name, showarrow=False, font={"size": 9, "color": "gray"},
            )
        fig.update_xaxes(title_text="Trial number (sequential)")
        fig.update_layout(
            title=f"Optuna XGBoost cross-validation: {scorer}",
            yaxis_title=f"Combined OOF {scorer}",
            template="plotly_white",
            hovermode="x unified",
        )
        history = run_dir / "optimization_history.html"
        fig.write_html(history, include_plotlyjs=True, full_html=True)
    except Exception as exc:
        raise ArtifactError(f"could not create optimization HTML: {exc}") from exc

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"round1": "tab:blue", "round2": "tab:orange", "single": "tab:blue"}
        incumbents = running_best(
            [float(v) for v in complete["penalized_score"].tolist()], direction
        )

        def _positive_for_log(values: list[float]) -> list[float]:
            positives = [float(value) for value in values if float(value) > 0]
            floor = min(positives) / 10 if positives else 1e-12
            return [float(value) if float(value) > 0 else floor for value in values]

        def _draw_progress(path: Path, *, log_y: bool) -> None:
            fig, ax = plt.subplots(figsize=(12, 7))
            for name in rounds:
                subset = complete[complete["round"] == name]
                scores = [float(value) for value in subset["penalized_score"].tolist()]
                ax.scatter(
                    subset["x"].tolist(),
                    _positive_for_log(scores) if log_y else scores,
                    label=f"{name} score",
                    color=colors.get(name, "tab:gray"),
                )
            ax.plot(
                complete["x"].tolist(), _positive_for_log(incumbents) if log_y else incumbents,
                label="running best", linewidth=2, color="black",
            )
            _iteration_axis(ax)
            ax.set(
                title=f"XGBoost combined OOF {scorer} (log scale)",
                ylabel=scorer,
            )
            if log_y:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        progress = run_dir / "optimization_progress.png"
        _draw_progress(progress, log_y=True)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"could not create optimization PNG: {exc}") from exc
    finally:
        shutil.rmtree(run_dir / ".matplotlib", ignore_errors=True)

    return trials_path, history, progress
