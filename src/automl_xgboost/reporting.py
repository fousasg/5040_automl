"""Full-row SHAP extraction and portable HTML explanation report."""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape

from .config import AutoMLConfig
from .data import PreparedData
from .exceptions import ArtifactError


def compute_shap_values(model: Any, data: PreparedData) -> tuple[Any, pd.DataFrame]:
    """Compute TreeSHAP on every row, retaining raw-margin additivity."""

    import shap

    try:
        explainer = shap.TreeExplainer(model, model_output="raw")
        explanation = explainer(data.features)
    except Exception as exc:
        raise ArtifactError(f"SHAP TreeExplainer failed: {exc}") from exc
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
        base_values = base_values[:, 1]
    if values.ndim != 2 or values.shape != data.features.shape:
        raise ArtifactError(
            f"unexpected SHAP shape {values.shape}; expected {data.features.shape}"
        )
    if base_values.ndim == 0:
        base_values = np.repeat(base_values, len(data.features))
    base_values = np.asarray(base_values).reshape(-1)
    if len(base_values) != len(data.features):
        raise ArtifactError("SHAP base-value row count does not match the dataset")
    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise ArtifactError("SHAP returned non-finite contributions")
    output = pd.DataFrame(values, columns=data.features.columns)
    output.insert(0, "row_id", data.row_ids)
    output["BiasTerm"] = base_values
    return explanation, output


def write_shap_report(
    explanation: Any,
    shap_frame: pd.DataFrame,
    data: PreparedData,
    config: AutoMLConfig,
    run_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, list[Path]]:
    """Create global/local SHAP plots and a movable HTML report."""

    import shap

    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets_dir = run_dir / "shap_assets"
    assets_dir.mkdir(exist_ok=True)
    indices = _sample_indices(data, config)
    display_data = data.features.iloc[indices].copy()
    for column in data.category_levels:
        display_data[column] = display_data[column].astype("string").fillna("<missing>")
    values = shap_frame.loc[indices, config.predictors].to_numpy()
    bases = shap_frame.loc[indices, "BiasTerm"].to_numpy()
    sampled = shap.Explanation(
        values=values,
        base_values=bases,
        data=display_data.to_numpy(),
        feature_names=config.predictors,
    )

    assets: list[Path] = []
    notes: list[str] = []
    plotters = [
        ("Global mean absolute SHAP importance", "importance.png", lambda: shap.plots.bar(sampled, show=False)),
        ("SHAP impact distribution", "beeswarm.png", lambda: shap.plots.beeswarm(sampled, show=False)),
        ("First sampled row explanation", "waterfall.png", lambda: shap.plots.waterfall(sampled[0], show=False)),
    ]
    sections: list[dict[str, str]] = []
    for title, filename, plotter in plotters:
        path = assets_dir / filename
        try:
            plt.close("all")
            plotter()
            figure = plt.gcf()
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            assets.append(path)
            sections.append({"title": title, "image": f"shap_assets/{filename}", "note": ""})
        except Exception as exc:
            plt.close("all")
            message = f"Plot unavailable: {exc}"
            notes.append(f"{title}: {exc}")
            sections.append({"title": title, "image": "", "note": message})

    importance = (
        shap_frame[config.predictors]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_absolute_shap")
        .rename_axis("feature")
        .reset_index()
    )
    missingness = pd.DataFrame(
        {
            "feature": config.predictors,
            "missing_rows": [data.missing_counts[name] for name in config.predictors],
            "missing_percent": [100 * data.missing_counts[name] / len(data.features) for name in config.predictors],
        }
    )
    environment = Environment(autoescape=select_autoescape(default=True))
    rendered = environment.from_string(_TEMPLATE).render(
        metadata_json=json.dumps(metadata, indent=2, sort_keys=True, default=str),
        sections=sections,
        importance_html=importance.to_html(index=False, escape=True, border=0),
        missingness_html=missingness.to_html(index=False, escape=True, border=0),
        explained_rows=len(indices),
        total_rows=len(data.features),
        notes=notes or ["All requested SHAP plots were generated."],
    )
    report = run_dir / "shap_report.html"
    temporary = run_dir / ".shap_report.html.tmp"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(report)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"could not write SHAP report: {exc}") from exc
    finally:
        shutil.rmtree(run_dir / ".matplotlib", ignore_errors=True)
    return report, assets


def _sample_indices(data: PreparedData, config: AutoMLConfig) -> np.ndarray:
    limit = config.explain_sample_size
    if limit is None or limit >= len(data.features):
        return np.arange(len(data.features))
    rng = np.random.default_rng(config.seed)
    if config.task == "regression":
        return np.sort(rng.choice(len(data.features), size=limit, replace=False))
    selected: list[int] = []
    y = data.encoded_target.to_numpy()
    for klass in (0, 1):
        candidates = np.flatnonzero(y == klass)
        quota = max(1, round(limit * len(candidates) / len(y)))
        selected.extend(rng.choice(candidates, size=min(quota, len(candidates)), replace=False).tolist())
    if len(selected) > limit:
        selected = rng.choice(selected, size=limit, replace=False).tolist()
    elif len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(y)), selected)
        selected.extend(rng.choice(remaining, size=limit - len(selected), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XGBoost SHAP report</title><style>
body{max-width:1200px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui,sans-serif;color:#17202a}
h1,h2{color:#123b5d}img{max-width:100%;height:auto;border:1px solid #d8dee4}pre{overflow:auto;background:#f6f8fa;padding:1rem}
table{border-collapse:collapse;display:block;overflow-x:auto}th,td{border:1px solid #d8dee4;padding:.4rem .7rem}th{background:#f6f8fa}.note{background:#fff8c5;padding:.7rem}
</style></head><body>
<h1>XGBoost SHAP explainability report</h1>
<p>Generated with <code>shap.TreeExplainer</code>. Contributions use XGBoost's raw margin scale. {{ explained_rows }} of {{ total_rows }} rows were sampled only for plots; the SHAP dataset contains every row.</p>
<h2>Run metadata</h2><pre>{{ metadata_json }}</pre>
<h2>Missing-value behavior</h2><p>Null predictor values were passed directly to XGBoost. No rows were dropped and no predictor values were imputed.</p>{{ missingness_html | safe }}
<h2>Global feature importance</h2>{{ importance_html | safe }}
{% for section in sections %}<h2>{{ section.title }}</h2>{% if section.image %}<img src="{{ section.image }}" alt="{{ section.title }}">{% else %}<p class="note">{{ section.note }}</p>{% endif %}{% endfor %}
<h2>Availability notes</h2><ul>{% for note in notes %}<li>{{ note }}</li>{% endfor %}</ul>
</body></html>"""

