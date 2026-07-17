from garxiv.retrieval.fusion import reciprocal_rank_fusion


def test_single_ranking_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [item for item, _ in fused] == ["a", "b", "c"]


def test_item_present_in_multiple_lists_scores_higher():
    fused = reciprocal_rank_fusion([
        ["a", "b", "c"],
        ["b", "a", "d"],
    ])
    order = [item for item, _ in fused]
    assert order[0] == "a"
    assert order[1] == "b"


def test_formula_matches_expected_values():
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert fused["a"] == 1.0 / 61
    assert fused["b"] == 1.0 / 62


def test_empty_rankings_return_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
