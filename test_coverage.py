from src.coverage import calibrate_threshold

PROBES = [{"topic": "attention", "best_score": 0.9}, {"topic": "retrieval", "best_score": 0.8},
          {"topic": "reinforcement-learning", "best_score": 0.2}]


def test_calibration_separates_known_topics():
    thr, acc, n = calibrate_threshold(PROBES, ["attention", "retrieval"], ["reinforcement-learning"])
    assert n == 3 and acc == 1.0 and 0.2 < thr <= 0.8


def test_calibration_without_labels():
    assert calibrate_threshold(PROBES, [], [])[2] == 0


from src.coverage import separability, topic_query, verdict_accuracy  # noqa: E402


def test_separability_is_auc():
    probes = [{"topic": "a", "best_score": 3.0}, {"topic": "b", "best_score": 2.0},
              {"topic": "x", "best_score": 1.0}]
    assert separability(probes, ["a", "b"], ["x"]) == 1.0
    assert separability(probes, ["x"], ["a", "b"]) == 0.0
    assert separability(probes, [], ["x"]) is None


def test_separability_exposes_the_majority_class_trap():
    # 5 present, 1 absent: a threshold below everything gets 5/6 right,
    # but the absent topic outscores 3 of the 5 present ones, so AUC < 0.5.
    probes = [{"topic": f"p{i}", "best_score": s} for i, s in enumerate([-3.7, 2.8, 2.0, 3.6, 4.2])]
    probes.append({"topic": "rl", "best_score": 3.5})
    present = [f"p{i}" for i in range(5)]
    thr, acc, n = calibrate_threshold(probes, present, ["rl"])
    assert abs(acc - 5 / 6) < 1e-9
    assert separability(probes, present, ["rl"]) == 0.4


def test_topic_query_prefers_supplied_phrase():
    assert topic_query("peft", {"peft": "parameter-efficient fine-tuning"}) == "parameter-efficient fine-tuning"
    assert topic_query("in-context-learning") == "in context learning"


def test_verdict_accuracy_counts_missing_as_wrong():
    rows = [{"topic": "a", "llm_covers": True}, {"topic": "x", "llm_covers": True},
            {"topic": "b", "llm_covers": None}, {"topic": "u", "llm_covers": True}]
    acc, n = verdict_accuracy(rows, ["a", "b"], ["x"])
    assert n == 3 and abs(acc - 1 / 3) < 1e-9
