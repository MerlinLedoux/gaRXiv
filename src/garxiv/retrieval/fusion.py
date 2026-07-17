def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Standard Reciprocal Rank Fusion. For each item, sums 1/(k + rank)
    over every ranked list it appears in. k=60 is the standard constant
    from the literature (Cormack et al. 2009) — it dampens the influence
    of low ranks without requiring score normalization across lists of
    different scales (e.g. cosine similarity vs bm25). Returns items
    sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
