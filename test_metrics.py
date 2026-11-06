from src.metrics import (aggregate, hit_at_k, precision_at_k, recall_at_k,
                         reciprocal_rank, score_question)

H = [{"doc": "a.pdf", "page": 3}, {"doc": "a.pdf", "page": 7}, {"doc": "b.pdf", "page": 1}]


def test_hit_at_k_respects_cutoff():
    assert hit_at_k(H, "a.pdf", [7], 1) == 0.0
    assert hit_at_k(H, "a.pdf", [7], 2) == 1.0


def test_hit_requires_matching_doc():
    assert hit_at_k(H, "b.pdf", [3], 5) == 0.0


def test_page_tolerance():
    assert hit_at_k(H, "a.pdf", [4], 5) == 0.0
    assert hit_at_k(H, "a.pdf", [4], 5, page_tol=1) == 1.0


def test_recall_counts_distinct_gold_pages():
    assert recall_at_k(H, "a.pdf", [3, 7], 3) == 1.0
    assert recall_at_k(H, "a.pdf", [3, 99], 3) == 0.5


def test_precision_and_mrr():
    assert precision_at_k(H, "a.pdf", [3], 3) == 1 / 3
    assert reciprocal_rank(H, "a.pdf", [7]) == 0.5
    assert reciprocal_rank(H, "zz.pdf", [1]) == 0.0


def test_aggregate_means():
    rows = [score_question(H, "a.pdf", [3]), score_question(H, "a.pdf", [99])]
    agg = aggregate(rows)
    assert agg["hit@1"] == 0.5 and agg["n"] == 2
