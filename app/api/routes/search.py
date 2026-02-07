from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline
from app.api.schemas import SearchRequest, SearchResponse, SourceDocument
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    chunks = await pipeline.search_only(
        query=request.query,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        metadata_filter=request.metadata_filter,
    )

    results = [
        SourceDocument(
            id=c.id,
            content=c.content,
            metadata=c.metadata,
            similarity=c.similarity,
            source_display=c.source_display,
        )
        for c in chunks
    ]

    return SearchResponse(
        results=results,
        query=request.query,
        count=len(results),
    )
