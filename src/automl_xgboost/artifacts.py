"""Safe run directories and machine-readable artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .config import AutoMLConfig
from .exceptions import ArtifactError, ConfigurationError

OWNED_FILES = {
    "manifest.json", "failure.json", "resolved_config.yaml", "run_metadata.json",
    "folds.csv", "folds.parquet", "trials.csv", "optimization_history.html",
    "optimization_progress.png", "cv_predictions.csv", "cv_predictions.parquet",
    "full_data_predictions.csv", "full_data_predictions.parquet", "shap_values.csv",
    "shap_values.parquet", "shap_report.html", "model.json", "model_metadata.json",
    "actual_vs_predicted.png", "confusion_matrix.png", "partial_dependence_top10.png",
    "optuna.db", "run.log",
}
OWNED_DIRECTORIES = {"shap_assets"}


def prepare_run_directory(config: AutoMLConfig, run_id: str) -> Path:
    requested = config.output_dir.expanduser().resolve()
    if not requested.exists():
        requested.mkdir(parents=True)
        return requested
    if not requested.is_dir():
        raise ConfigurationError(f"output_dir is not a directory: {requested}")
    if not any(requested.iterdir()):
        return requested
    if config.overwrite:
        if requested == requested.parent or len(requested.parts) < 3:
            raise ConfigurationError(f"refusing unsafe overwrite: {requested}")
        for name in OWNED_FILES:
            path = requested / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        for name in OWNED_DIRECTORIES:
            path = requested / name
            if path.is_dir():
                shutil.rmtree(path)
        return requested
    child = requested / run_id
    child.mkdir()
    return child


def configure_logging(run_dir: Path, run_id: str) -> tuple[logging.Logger, Path]:
    path = run_dir / "run.log"
    logger = logging.getLogger(f"automl_xgboost.{run_id}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s run_id=%(run_id)s trial=%(trial_number)s %(message)s",
            defaults={"run_id": run_id, "trial_number": "-"},
        )
    )
    logger.addHandler(handler)
    return logger, path


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def atomic_text(path: Path, text: str) -> Path:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"could not write {path.name}: {exc}") from exc
    return path


def atomic_json(path: Path, value: Any) -> Path:
    return atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_yaml(path: Path, value: Any) -> Path:
    return atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def write_dataframe(frame: pd.DataFrame, base: Path, output_format: str) -> Path:
    path = base.with_suffix(f".{output_format}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        if output_format == "parquet":
            frame.to_parquet(temporary, index=False, engine="pyarrow")
        else:
            frame.to_csv(temporary, index=False)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"could not write {path.name}: {exc}") from exc
    return path


def write_manifest(
    run_dir: Path, artifacts: Iterable[tuple[Path, str, int | None]], run_id: str
) -> Path:
    entries = []
    for path, media_type, row_count in artifacts:
        if not path.is_file():
            raise ArtifactError(f"required artifact is missing: {path}")
        entry = {
            "path": str(path.relative_to(run_dir)), "media_type": media_type,
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
        }
        if row_count is not None:
            entry["row_count"] = row_count
        entries.append(entry)
    return atomic_json(
        run_dir / "manifest.json",
        {
            "status": "success", "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": sorted(entries, key=lambda item: item["path"]),
        },
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
