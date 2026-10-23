from pathlib import Path

from automl_xgboost.cli import build_parser, resolve_cli_config
from automl_xgboost.data import load_dataset, prepare_data


def test_cli_overrides_config_and_resolves_relative_paths(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n2,3\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset: data.csv\ntarget: y\ntask: regression\nscorer: rmse\n"
        "predictors: [x]\nn_trials: 10\noutput_dir: output\n"
    )
    args = build_parser().parse_args(["run", "--config", str(config), "--n-trials", "3"])
    resolved = resolve_cli_config(args)
    assert resolved.n_trials == 3
    assert resolved.dataset == data.resolve()
    assert resolved.output_dir == (tmp_path / "output").resolve()


def test_included_classification_config_resolves_positive_class():
    config_path = Path(__file__).parents[2] / "examples" / "classification.yaml"
    args = build_parser().parse_args(["run", "--config", str(config_path)])
    resolved = resolve_cli_config(args)
    prepared = prepare_data(load_dataset(resolved.dataset), resolved)
    assert prepared.class_levels == ["no", "yes"]


def test_score_subcommand_contract(tmp_path):
    args = build_parser().parse_args([
        "score", "--run-dir", str(tmp_path), "--dataset", "new.csv",
        "--output", "predictions.parquet",
    ])
    assert args.command == "score"
    assert args.run_dir == tmp_path
