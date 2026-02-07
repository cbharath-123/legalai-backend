from functools import lru_cache

from app.llm.factory import create_llm_provider
from app.rag.pipeline import RAGPipeline


@lru_cache
def get_pipeline() -> RAGPipeline:
    provider = create_llm_provider()
    return RAGPipeline(llm_provider=provider)
