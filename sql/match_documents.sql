-- =============================================================================
-- Deploy this file in the Supabase SQL Editor.
-- It creates:
--   1. An HNSW index on documents2.embedding for fast cosine similarity
--   2. A match_documents() RPC function for pgvector search
-- =============================================================================

-- 1. HNSW Index (better recall and query performance than ivfflat)
--    m = 16: max connections per node (higher = better recall, more memory)
--    ef_construction = 64: build-time search width (higher = better index, slower build)
--
--    NOTE: For large tables (>200K rows), run these SET commands first:
--      SET maintenance_work_mem = '512MB';
--      SET statement_timeout = 0;
DROP INDEX IF EXISTS idx_documents2_embedding_cosine;

CREATE INDEX idx_documents2_embedding_cosine
ON documents2
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);


-- 2. RPC function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 8,
    filter jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents2 d
    WHERE 1 - (d.embedding <=> query_embedding) >= match_threshold
      AND (filter = '{}'::jsonb OR d.metadata @> filter)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
