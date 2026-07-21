"""Unit tests for the pure decision functions in modeling/processors.py.

These two functions decide which rows survive into the model and how scaffold
levels are labelled -- a wrong branch silently drops a benchmark or splits a
scaffold, which no downstream check catches. The end-to-end pipeline test is
modeling/synth/run_synth.py; everything else in processors.py is glue over
hibayes and is exercised there.
"""

import math

import pytest

from modeling.processors import _coerce_one, canonical_args


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Inspect letter grades
        ("C", 1.0),
        ("I", 0.0),
        ("N", 0.0),
        ("P", 0.5),  # partial credit: coerced, then dropped as non-binary
        (" C ", 1.0),
        # JSONB booleans and their string forms
        (True, 1.0),
        (False, 0.0),
        ("true", 1.0),
        ("False", 0.0),
        # numerics
        (1, 1.0),
        (0.0, 0.0),
        ("1", 1.0),
        ("0.5", 0.5),
        # unparseable / missing -> None (dropped)
        (None, None),
        (float("nan"), None),
        ("garbage", None),
        ("", None),
    ],
)
def test_coerce_one(value: object, expected: float | None) -> None:
    result = _coerce_one(value)
    if expected is None:
        assert result is None
    else:
        assert result is not None and math.isclose(result, expected)


def test_canonical_args_is_order_stable() -> None:
    assert canonical_args('{"b": 2, "a": 1}') == canonical_args('{"a": 1, "b": 2}')


def test_canonical_args_exclude_mode() -> None:
    raw = '{"max_attempts": 3, "variants": "hard"}'
    assert canonical_args(raw, excluded={"variants"}) == "max_attempts=3"


def test_canonical_args_allowlist_overrides_exclude() -> None:
    raw = '{"max_attempts": 3, "variants": "hard", "seed": 7}'
    assert (
        canonical_args(raw, excluded={"variants"}, included={"max_attempts"})
        == "max_attempts=3"
    )


def test_canonical_args_missing_and_empty() -> None:
    assert canonical_args(None) == ""
    assert canonical_args(float("nan")) == ""
    assert canonical_args("{}") == ""
