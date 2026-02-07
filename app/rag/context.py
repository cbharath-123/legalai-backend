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

# Common German legal compound words broken by OCR
_OCR_WORD_FIXES = [
    # Legal terms
    (re.compile(r"\bstraf\s+recht\s*lich\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bHaft\s+ung\b", re.IGNORECASE), "Haftung"),
    (re.compile(r"\bOr\s*dn\s*ungs\s*wid\s*rig\s*keit\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bStra\s*f\s*t\s*at\s*en?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bV\s*ors\s*atz\b", re.IGNORECASE), "Vorsatz"),
    (re.compile(r"\bF\s*ah\s*rl\s*äss\s*ig\s*keit\b", re.IGNORECASE), "Fahrlässigkeit"),
    (re.compile(r"\bTat\s+bestand\b", re.IGNORECASE), "Tatbestand"),
    (re.compile(r"\bRe\s*chts\s*wid\s*rig\s*keit\b", re.IGNORECASE), "Rechtswidrigkeit"),
    (re.compile(r"\bSch\s*uld\s*un\s*fähig\s*keit\b", re.IGNORECASE), "Schuldunfähigkeit"),
    (re.compile(r"\bSch\s+uld\b", re.IGNORECASE), "Schuld"),
    (re.compile(r"\bStr\s*af\s*bar\s*keit\b", re.IGNORECASE), "Strafbarkeit"),
    (re.compile(r"\bStr\s+af\s+maß\b", re.IGNORECASE), "Strafmaß"),
    (re.compile(r"\bRechts\s*anw\s*alt\b", re.IGNORECASE), "Rechtsanwalt"),
    (re.compile(r"\bVertr\s*äge?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bVerein\s*barung\b", re.IGNORECASE), "Vereinbarung"),
    (re.compile(r"\bgesetz\s*buch\b", re.IGNORECASE), "gesetzbuch"),
    (re.compile(r"\bRecht\s*fert\s*igungs\s*grund\b", re.IGNORECASE), "Rechtfertigungsgrund"),
    (re.compile(r"\bNot\s*wehr\b", re.IGNORECASE), "Notwehr"),
    (re.compile(r"\bNot\s*stand\b", re.IGNORECASE), "Notstand"),
    (re.compile(r"\bBe\s*geh\s*ung\b", re.IGNORECASE), "Begehung"),
    (re.compile(r"\bTä\s*ters?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bEins\s*icht\b", re.IGNORECASE), "Einsicht"),
    (re.compile(r"\bver\s*wirk\s*lich\s*ung\b", re.IGNORECASE), "Verwirklichung"),
    (re.compile(r"\bwirk\s+lich\b", re.IGNORECASE), "wirklich"),
    (re.compile(r"\bS\s*org\s*falt\b", re.IGNORECASE), "Sorgfalt"),
    (re.compile(r"\bUm\s*stände\b", re.IGNORECASE), "Umstände"),
    (re.compile(r"\bAs\s*pekt\b", re.IGNORECASE), "Aspekt"),
    (re.compile(r"\bspezif\s*isch\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bdetaill\s*iert\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bStra\s*f\s*tat\b", re.IGNORECASE), "Straftat"),
    (re.compile(r"\bdefini\s*ert\b", re.IGNORECASE), "definiert"),
    (re.compile(r"\bgehand\s*elt\b", re.IGNORECASE), "gehandelt"),
    (re.compile(r"\bschuld\s*haft\b", re.IGNORECASE), "schuldhaft"),
    (re.compile(r"\bSch\s*were?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bpers\s*önlich\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bgest\s*ellt\w*\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bAllgeme\s*ine?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bGrundlagen\b", re.IGNORECASE), "Grundlagen"),
    (re.compile(r"\bwid\s*rig\b", re.IGNORECASE), "widrig"),
    (re.compile(r"\bStra\s*ft\s*aten\b", re.IGNORECASE), "Straftaten"),
    (re.compile(r"\bverm\s*inder\s*te?\b", re.IGNORECASE), lambda m: m.group().replace(" ", "")),
    (re.compile(r"\bähn\s*igkeit\b", re.IGNORECASE), "ähnigkeit"),
    (re.compile(r"\bSt\s+ra\s*f\b"), "Straf"),
    (re.compile(r"\bRe\s+chts\b"), "Rechts"),
    (re.compile(r"\bW\s+ollen\b"), "Wollen"),
]



def _rejoin_broken_words(text: str) -> str:
    """
    Heuristic: rejoin single-letter or short fragments that are clearly
    parts of one word split by OCR. E.g. "V ors atz" → "Vorsatz".
    This handles patterns where a single letter is followed by a space
    and then a lowercase continuation.
    """
    # Join: uppercase letter + space + lowercase fragment(s)
    # e.g. "V ors atz" or "F ah rl äss ig keit"
    text = re.sub(
        r'\b([A-ZÄÖÜ])\s+([a-zäöüß]{1,4})(?=\s+[a-zäöüß])',
        r'\1\2',
        text,
    )
    # Repeated pass to catch remaining fragments
    for _ in range(5):
        new_text = re.sub(
            r'\b([A-ZÄÖÜa-zäöüß]{2,})\s+([a-zäöüß]{1,4})\b(?=\s|[.,;:!?)]|$)',
            r'\1\2',
            text,
        )
        if new_text == text:
            break
        text = new_text
    return text


def clean_chunk_text(text: str) -> str:
    """
    Clean common OCR artifacts from document chunk text.

    Fixes broken legal abbreviations, broken German compound words,
    collapses multiple spaces, and removes spaces before punctuation.
    """
    # Fix legal abbreviations
    for pattern, replacement in _LEGAL_ABBREV_FIXES:
        text = pattern.sub(replacement, text)

    # Fix known broken German legal words
    for pattern, replacement in _OCR_WORD_FIXES:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)

    # General heuristic: rejoin fragments split by OCR
    text = _rejoin_broken_words(text)

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
