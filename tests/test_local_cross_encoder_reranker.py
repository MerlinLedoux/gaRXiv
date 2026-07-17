import pytest


@pytest.mark.slow
def test_local_cross_encoder_reranker_scores_relevant_pair_higher():
    from garxiv.rerank.local_cross_encoder import LocalCrossEncoderReranker

    reranker = LocalCrossEncoderReranker()
    scores = reranker.score(
        "What is the Turing test?",
        [
            "The Turing test evaluates whether a machine exhibits intelligent behavior.",
            "Bananas are a good source of potassium.",
        ],
    )

    assert len(scores) == 2
    assert scores[0] > scores[1]


@pytest.mark.slow
def test_local_cross_encoder_reranker_handles_empty_texts():
    from garxiv.rerank.local_cross_encoder import LocalCrossEncoderReranker

    reranker = LocalCrossEncoderReranker()
    assert reranker.score("query", []) == []
