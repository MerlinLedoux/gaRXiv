import pytest


@pytest.mark.slow
def test_local_bge_provider_embeds_with_expected_dimension():
    from garxiv.embeddings.local_bge import LocalBgeEmbeddingProvider

    provider = LocalBgeEmbeddingProvider()
    vectors = provider.embed(["the cat sat on the mat", "quantum field theory"])

    assert len(vectors) == 2
    assert len(vectors[0]) == provider.dimension == 384


@pytest.mark.slow
def test_local_bge_provider_similar_sentences_score_higher():
    import numpy as np

    from garxiv.embeddings.local_bge import LocalBgeEmbeddingProvider

    provider = LocalBgeEmbeddingProvider()
    a, b, c = provider.embed([
        "The transformer attention mechanism computes weighted sums.",
        "Self-attention in transformers uses weighted combinations.",
        "Bananas are a good source of potassium.",
    ])

    sim_related = float(np.dot(a, b))
    sim_unrelated = float(np.dot(a, c))
    assert sim_related > sim_unrelated
