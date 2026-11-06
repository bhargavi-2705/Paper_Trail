from src.judge import cohens_kappa, validate_judge


def test_kappa_perfect_and_chance():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_validate_judge_thresholds_scores():
    recs = [{"judge_score": 0.9, "human_label": 1}, {"judge_score": 0.2, "human_label": 0},
            {"judge_score": 0.8, "human_label": 0}]
    out = validate_judge(recs, threshold=0.7)
    assert out["n"] == 3 and abs(out["agreement"] - 2 / 3) < 1e-9


def test_validate_judge_ignores_unlabelled():
    assert validate_judge([{"judge_score": 0.9, "human_label": None}])["n"] == 0


# --- regression: LaTeX backslashes in judge replies made json.loads fail ------
from src.judge import parse_json  # noqa: E402


def test_parse_json_survives_latex_backslashes():
    raw = '{"claims":[{"claim":"divide by \\( \\sqrt{d_k} \\)","supported":true}],"score":1.0}'
    assert parse_json(raw, None)["score"] == 1.0


def test_parse_json_handles_fences_and_chatter():
    raw = 'Sure!\n```json\n{"claims":[],"score":0.5}\n```\nHope this helps'
    assert parse_json(raw, None)["score"] == 0.5


def test_parse_json_falls_back_on_garbage():
    assert parse_json("no json here", {"score": None}) == {"score": None}
    assert parse_json('{"claims": [unterminated', {"score": None}) == {"score": None}


def test_parse_json_does_not_mangle_already_escaped_latex():
    """Valid JSON with escaped backslashes must come through unchanged."""
    raw = '{"claims":[{"claim":"divide by \\\\( x \\\\)","supported":true}],"score":1.0}'
    assert parse_json(raw, None)["claims"][0]["claim"] == "divide by \\( x \\)"


def test_parse_json_handles_mixed_escaped_and_unescaped():
    raw = '{"claims":[{"claim":"\\\\alpha and \\beta","supported":true}],"score":1.0}'
    assert parse_json(raw, None) is not None
