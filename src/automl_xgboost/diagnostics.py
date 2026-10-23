"""Validation-performance and partial-dependence plots for the winning model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AutoMLConfig
from .data import PreparedData
from .exceptions import ArtifactError
from .modeling import predict_for_score


def write_performance_plot(
    data: PreparedData,
    oof_predictions: np.ndarray,
    config: AutoMLConfig,
    run_dir: Path,
) -> Path:
    """Plot OOF actual-versus-predicted values or an OOF confusion matrix."""

    predictions = np.asarray(oof_predictions, dtype=float)
    if len(predictions) != len(data.features) or not np.isfinite(predictions).all():
        raise ArtifactError("performance plot requires one finite OOF prediction per row")

    plt = _pyplot(run_dir)
    figure, axis = plt.subplots(figsize=(8, 7))
    try:
        if config.task == "regression":
            actual = data.encoded_target.to_numpy(dtype=float)
            lower = float(min(actual.min(), predictions.min()))
            upper = float(max(actual.max(), predictions.max()))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            axis.scatter(actual, predictions, alpha=0.65, edgecolors="none")
            axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1.5)
            axis.set(
                title="Out-of-fold actual vs predicted",
                xlabel="Actual",
                ylabel="Out-of-fold prediction",
                xlim=(lower, upper),
                ylim=(lower, upper),
            )
            axis.set_aspect("equal", adjustable="box")
            filename = "actual_vs_predicted.png"
        else:
            actual = data.encoded_target.to_numpy(dtype=np.int8)
            predicted = (predictions >= 0.5).astype(np.int8)
            matrix = np.zeros((2, 2), dtype=np.int64)
            np.add.at(matrix, (actual, predicted), 1)
            image = axis.imshow(matrix, cmap="Blues")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            levels = [str(value) for value in (data.class_levels or [0, 1])]
            axis.set(
                title="Out-of-fold confusion matrix (threshold = 0.5)",
                xlabel="Predicted class",
                ylabel="Actual class",
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=levels,
                yticklabels=levels,
            )
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in range(2):
                for column in range(2):
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:,}",
                        ha="center",
                        va="center",
                        color="white" if matrix[row, column] > threshold else "black",
                    )
            filename = "confusion_matrix.png"
        if config.task == "classification":
            axis.grid(False)
        else:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        path = run_dir / filename
        _save_png(figure, path)
        return path
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"could not create model performance plot: {exc}") from exc
    finally:
        plt.close(figure)


def write_partial_dependence_plot(
    model: Any,
    data: PreparedData,
    config: AutoMLConfig,
    run_dir: Path,
    *,
    max_features: int = 10,
    grid_points: int = 20,
    sample_size: int = 1_000,
) -> tuple[Path, list[str]]:
    """Plot one-way PDPs for the most important fitted-model predictors."""

    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if importances.shape != (len(config.predictors),) or not np.isfinite(importances).all():
        raise ArtifactError("XGBoost returned invalid feature importances for PDP ranking")
    feature_order = sorted(
        range(len(config.predictors)),
        key=lambda index: (-float(importances[index]), index),
    )[: min(max_features, len(config.predictors))]
    selected = [config.predictors[index] for index in feature_order]
    if not selected:
        raise ArtifactError("at least one predictor is required for partial dependence")

    background = _sample_features(data.features, sample_size, config.seed)
    columns = 2 if len(selected) > 1 else 1
    rows = (len(selected) + columns - 1) // columns
    plt = _pyplot(run_dir)
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4 * rows), squeeze=False)
    try:
        for panel, feature in enumerate(selected):
            axis = axes.flat[panel]
            importance = float(importances[config.predictors.index(feature)])
            x_values, averages, categorical = _partial_dependence(
                model, background, feature, data, config, grid_points
            )
            if not x_values:
                axis.text(0.5, 0.5, "No observed values", ha="center", va="center")
            elif categorical:
                positions = np.arange(len(x_values))
                axis.bar(positions, averages)
                axis.set_xticks(positions, x_values, rotation=45, ha="right")
            else:
                axis.plot(np.asarray(x_values, dtype=float), averages, marker="o", markersize=3)
            axis.set_title(f"{feature} (importance={importance:.4g})")
            axis.set_xlabel(feature)
            axis.set_ylabel(
                "Mean predicted probability" if config.task == "classification" else "Mean prediction"
            )
            axis.grid(alpha=0.25)
        for panel in range(len(selected), len(axes.flat)):
            axes.flat[panel].set_visible(False)
        figure.suptitle("Partial dependence for top XGBoost features", fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = run_dir / "partial_dependence_top10.png"
        _save_png(figure, path)
        return path, selected
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"could not create partial-dependence plot: {exc}") from exc
    finally:
        plt.close(figure)


def _partial_dependence(
    model: Any,
    background: pd.DataFrame,
    feature: str,
    data: PreparedData,
    config: AutoMLConfig,
    grid_points: int,
) -> tuple[list[Any], np.ndarray, bool]:
    series = data.features[feature]
    if feature in data.category_levels:
        observed = series.value_counts(dropna=True)
        x_values = [str(value) for value in observed.index[:grid_points]]
        categorical = True
    else:
        observed = series.dropna().to_numpy(dtype=float)
        if not len(observed):
            return [], np.asarray([], dtype=float), False
        unique = np.unique(observed)
        if len(unique) <= grid_points:
            x_values = unique.tolist()
        else:
            x_values = np.unique(
                np.quantile(observed, np.linspace(0.05, 0.95, grid_points))
            ).tolist()
        categorical = False

    averages = []
    for value in x_values:
        modified = background.copy()
        if categorical:
            categories = data.features[feature].cat.categories
            modified[feature] = pd.Categorical(
                [value] * len(modified), categories=categories
            )
        else:
            modified[feature] = float(value)
        prediction = predict_for_score(model, modified, config.task)
        if len(prediction) != len(modified) or not np.isfinite(prediction).all():
            raise ArtifactError(f"PDP prediction failed for predictor {feature!r}")
        averages.append(float(np.mean(prediction)))
    return x_values, np.asarray(averages, dtype=float), categorical


def _sample_features(features: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(features) <= sample_size:
        return features.copy()
    return features.sample(n=sample_size, random_state=seed).sort_index().copy()


def _pyplot(run_dir: Path) -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_png(figure: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"could not write {path.name}: {exc}") from exc
