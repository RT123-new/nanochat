import ast
from pathlib import Path

import pytest


def _load_normalize_random_baseline():
    source = Path("scripts/base_eval.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    fn_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_random_baseline"
    )
    fn_module = ast.Module(body=[fn_node], type_ignores=[])
    namespace = {}
    exec(compile(fn_module, filename="scripts/base_eval.py", mode="exec"), namespace)
    return namespace["normalize_random_baseline"]


normalize_random_baseline = _load_normalize_random_baseline()


def test_normalize_random_baseline_percentage_input():
    assert normalize_random_baseline("25") == pytest.approx(0.25)


def test_normalize_random_baseline_fraction_input():
    assert normalize_random_baseline("0.25") == pytest.approx(0.25)


def test_normalize_random_baseline_negative_rejected():
    with pytest.raises(ValueError):
        normalize_random_baseline("-0.1")


def test_normalize_random_baseline_one_or_higher_rejected_after_normalization():
    with pytest.raises(ValueError):
        normalize_random_baseline("1.0")
