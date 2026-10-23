from pathlib import Path


def test_source_contains_no_h2o_dependency_or_import():
    content = "\n".join(path.read_text().lower() for path in Path("src").rglob("*.py"))
    assert "h2o" not in content


def test_requirements_contain_xgboost_and_shap_but_not_h2o():
    content = Path("requirements.txt").read_text().lower()
    assert "xgboost" in content and "shap" in content
    assert "h2o" not in content

