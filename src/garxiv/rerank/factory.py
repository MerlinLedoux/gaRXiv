from garxiv.config import RerankConfig
from garxiv.rerank.base import Reranker


def get_reranker(config: RerankConfig) -> Reranker:
    if config.provider == "local_cross_encoder":
        from garxiv.rerank.local_cross_encoder import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(model_name=config.model_name, device=config.device)

    raise ValueError(f"unknown reranker provider: {config.provider}")
