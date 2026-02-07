from __future__ import annotations

import re
import tiktoken
import structlog

from app.core.config import get_settings
from app.db.models import DocumentChunk
from app.rag.prompts import CONTEXT_BLOCK_TEMPLATE, CONTEXT_TEMPLATE

logger = structlog.get_logger(__name__)

# Common German legal abbreviations with OCR spacing errors
_LEGAL_ABBREV_FIXES = [
    (re.compile(r"\bB\s+GB\b"), "BGB"),
    (re.compile(r"\bSt\s+GB\b"), "StGB"),
    (re.compile(r"\bH\s+GB\b"), "HGB"),
    (re.compile(r"\bG\s+G\b"), "GG"),
    (re.compile(r"\bZ\s+PO\b"), "ZPO"),
    (re.compile(r"\bSt\s+PO\b"), "StPO"),
    (re.compile(r"\bVw\s+VfG\b"), "VwVfG"),
    (re.compile(r"\bVw\s+GO\b"), "VwGO"),
    (re.compile(r"\bBG\s+B\b"), "BGB"),
    (re.compile(r"\bAbs\s+\."), "Abs."),
    (re.compile(r"\bNr\s+\."), "Nr."),
    (re.compile(r"\bArt\s+\."), "Art."),
]



def clean_chunk_text(text: str) -> str:
    """
    Clean common OCR artifacts from document chunk text.

    Fixes broken legal abbreviations, collapses multiple spaces,
    and removes spaces before punctuation.
    """
    # Fix legal abbreviations
    for pattern, replacement in _LEGAL_ABBREV_FIXES:
        text = pattern.sub(replacement, text)

    # Collapse multiple spaces to single
    text = re.sub(r"  +", " ", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)

    return text.strip()

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def assemble_context(
    chunks: list[DocumentChunk],
    max_tokens: int | None = None,
) -> str:
    """
    Build a context string from ranked chunks, respecting a token budget.

    Greedily adds chunks in similarity order until the budget is exhausted.
    """
    settings = get_settings()
    max_tokens = max_tokens or settings.max_context_tokens

    blocks: list[str] = []
    tokens_used = 0

    for i, chunk in enumerate(chunks, start=1):
        block = CONTEXT_BLOCK_TEMPLATE.format(
            index=i,
            similarity=chunk.similarity,
            source=chunk.source_display,
            content=clean_chunk_text(chunk.content),
        )
        block_tokens = count_tokens(block)

        if tokens_used + block_tokens > max_tokens:
            logger.info(
                "context_budget_reached",
                chunks_included=i - 1,
                tokens_used=tokens_used,
                budget=max_tokens,
            )
            break

        blocks.append(block)
        tokens_used += block_tokens

    if not blocks:
        return ""

    context = CONTEXT_TEMPLATE.format(context_blocks="\n".join(blocks))
    logger.info("context_assembled", chunks=len(blocks), tokens=tokens_used)
    return context
