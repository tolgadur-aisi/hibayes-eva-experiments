"""Unit tests for discovery's outcome classification.

Regression suite for a real production failure: eva expands the full Inspect
score object into several score_* columns (value, answer, explanation), and
pooling them all misclassified every real benchmark as "other". These tests
pin the column-selection and classification decisions on realistic shapes.
"""

import pandas as pd

from modeling.discovery.list_evals import classify_score_columns, classify_values


def test_headline_column_wins_over_polluted_answer_columns() -> None:
    # the cybench shape: letter values + flag-string answers + explanations
    df = pd.DataFrame(
        {
            "score_includes": ["C", "I", "I", "C"],
            "score_includes.answer": ["htb{flag_one}", "htb{flag_two}", None, "x"],
            "score_includes.explanation": ["No text."] * 4,
        }
    )
    col, outcome, top = classify_score_columns(df, "includes")
    assert col == "score_includes"
    assert outcome == "pass_fail"
    assert top == {"C": 2, "I": 2}


def test_float_binary_scores_classify_pass_fail() -> None:
    # the swe_bench shape: 0.0/1.0 values plus metadata dumps
    df = pd.DataFrame(
        {
            "score_swe_bench_scorer": [0.0, 1.0, 0.0],
            "score_swe_bench_scorer.answer": ['{"model_patch": ""}'] * 3,
        }
    )
    col, outcome, _ = classify_score_columns(df, "swe_bench_scorer")
    assert col == "score_swe_bench_scorer"
    assert outcome == "pass_fail"


def test_no_headline_falls_back_to_most_decision_like_column() -> None:
    df = pd.DataFrame(
        {
            "score_grader": ["some text", "other text"],
            "score_checker": ["C", "I"],
        }
    )
    col, outcome, _ = classify_score_columns(df, None)
    assert col == "score_checker"
    assert outcome == "pass_fail"


def test_headline_with_only_dotted_expansion_uses_best_of_them() -> None:
    df = pd.DataFrame(
        {
            "score_rubric.value": [1, 0, 1],
            "score_rubric.explanation": ["free text", "more text", "words"],
        }
    )
    col, outcome, _ = classify_score_columns(df, "rubric")
    assert col == "score_rubric.value"
    assert outcome == "pass_fail"


def test_dict_cells_are_skipped_not_fatal() -> None:
    df = pd.DataFrame({"score_x": [{"nested": 1}, "C", "I"]})
    col, outcome, _ = classify_score_columns(df, None)
    assert col == "score_x"
    assert outcome == "pass_fail"


def test_no_score_columns() -> None:
    df = pd.DataFrame({"other_col": [1, 2]})
    assert classify_score_columns(df, "includes") == (None, "no_scores", {})


def test_classify_values_string_floats() -> None:
    from collections import Counter

    assert classify_values(Counter(["0.0", "1.0", "0.0"])) == "pass_fail"
