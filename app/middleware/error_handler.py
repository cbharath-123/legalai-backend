from fastapi import Request
from fastapi.responses import JSONResponse
import structlog

from app.core.exceptions import (
    DatabaseConnectionError,
    EmbeddingError,
    LLMGenerationError,
    LLMProviderNotFoundError,
    RAGBaseError,
    VectorSearchError,
)

logger = structlog.get_logger(__name__)

_STATUS_MAP = {
    EmbeddingError: 502,
    VectorSearchError: 502,
    LLMGenerationError: 502,
    LLMProviderNotFoundError: 400,
    DatabaseConnectionError: 503,
}


async def rag_exception_handler(request: Request, exc: RAGBaseError) -> JSONResponse:
    status_code = _STATUS_MAP.get(type(exc), 500)
    logger.error(
        "request_error",
        error_type=type(exc).__name__,
        message=exc.message,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.message},
    )
