from __future__ import annotations

import asyncpg
import structlog

from app.core.config import get_settings
from app.core.exceptions import DatabaseConnectionError

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None or _pool._closed:
        settings = get_settings()
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            logger.info("database_pool_created")
        except Exception as exc:
            logger.error("database_pool_creation_failed", error=str(exc))
            raise DatabaseConnectionError(str(exc)) from exc
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None and not _pool._closed:
        await _pool.close()
        _pool = None
        logger.info("database_pool_closed")


async def check_db_connection() -> bool:
    """Ping the database. Returns True if healthy."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False
