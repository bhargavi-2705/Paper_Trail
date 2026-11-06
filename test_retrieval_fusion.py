from src.retrieve import reciprocal_rank_fusion


def _lst(ids):
    return [{"id": i, "text": i, "doc": "d", "page": 1, "score": 0.0} for i in ids]


def test_rrf_rewards_agreement_across_lists():
    """b is 2nd in both lists; a and z are 1st in only one. Agreement wins."""
    fused = reciprocal_rank_fusion([_lst(["a", "b", "c"]), _lst(["z", "b", "y"])], k=60)
    assert fused[0]["id"] == "b"


def test_rrf_dedupes_and_keeps_all():
    fused = reciprocal_rank_fusion([_lst(["a", "b"]), _lst(["b", "a"])], k=60)
    assert sorted(h["id"] for h in fused) == ["a", "b"]


def test_rrf_scores_descend():
    fused = reciprocal_rank_fusion([_lst(["a", "b", "c"])], k=60)
    assert [h["score"] for h in fused] == sorted((h["score"] for h in fused), reverse=True)


def test_small_k_sharpens_rank_weighting():
    """k controls how steeply score falls with rank: smaller k = bigger gap
    between 1st and 2nd. (Agreement across lists still wins at any k -- two
    2nd places always outscore one 1st place.)"""
    sharp = reciprocal_rank_fusion([_lst(["a", "b"])], k=1)
    flat = reciprocal_rank_fusion([_lst(["a", "b"])], k=60)
    assert sharp[0]["score"] / sharp[1]["score"] > flat[0]["score"] / flat[1]["score"]
