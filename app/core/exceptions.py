class RAGBaseError(Exception):
    """Base exception for the RAG pipeline."""

    def __init__(self, message: str = "An internal error occurred."):
        self.message = message
        super().__init__(self.message)


class EmbeddingError(RAGBaseError):
    """Failed to generate embeddings."""

    def __init__(self, message: str = "Failed to generate query embedding."):
        super().__init__(message)


class VectorSearchError(RAGBaseError):
    """Failed to perform vector similarity search."""

    def __init__(self, message: str = "Vector search failed."):
        super().__init__(message)


class LLMGenerationError(RAGBaseError):
    """LLM failed to generate a response."""

    def __init__(self, message: str = "LLM generation failed."):
        super().__init__(message)


class LLMProviderNotFoundError(RAGBaseError):
    """Requested LLM provider is not available."""

    def __init__(self, provider: str):
        super().__init__(f"LLM provider '{provider}' is not supported.")


class DatabaseConnectionError(RAGBaseError):
    """Cannot connect to the database."""

    def __init__(self, message: str = "Database connection failed."):
        super().__init__(message)


class ContextAssemblyError(RAGBaseError):
    """Failed to assemble context for the LLM."""

    def __init__(self, message: str = "Context assembly failed."):
        super().__init__(message)
