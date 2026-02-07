from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import structlog

from app.core.config import get_settings
from app.db.models import ConversationMessage, DocumentChunk
from app.db.vector_search import search_documents
from app.llm.base import BaseLLMProvider
from app.llm.embeddings import embed_query
from app.rag.context import assemble_context
from app.rag.prompts import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


def _fix_ocr_words_in_output(text: str) -> str:
    """Fix common OCR-broken German words that the LLM may have copied."""
    fixes = [
        (r'\bstraf\s+recht\s*lich\w*', lambda m: m.group().replace(' ', '')),
        (r'\bHaft\s+ung', 'Haftung'),
        (r'\bV\s*ors\s*atz', 'Vorsatz'),
        (r'\bF\s*ah\s*rl\s*äss\s*ig\s*keit', 'Fahrlässigkeit'),
        (r'\bTat\s+bestand', 'Tatbestand'),
        (r'\bRe\s*chts\s*wid\s*rig\s*keit', 'Rechtswidrigkeit'),
        (r'\bSch\s+uld', 'Schuld'),
        (r'\bStr\s*af\s*bar\s*keit', 'Strafbarkeit'),
        (r'\bStr\s+af\s+maß', 'Strafmaß'),
        (r'\bRechts\s*anw\s*alt', 'Rechtsanwalt'),
        (r'\bVerein\s*barung', 'Vereinbarung'),
        (r'\bRecht\s*fert\s*igungs\s*grund', 'Rechtfertigungsgrund'),
        (r'\bNot\s+wehr', 'Notwehr'),
        (r'\bNot\s+stand', 'Notstand'),
        (r'\bBe\s*geh\s*ung', 'Begehung'),
        (r'\bver\s*wirk\s*lich\s*ung', 'Verwirklichung'),
        (r'\bS\s*org\s*falt', 'Sorgfalt'),
        (r'\bUm\s*stände', 'Umstände'),
        (r'\bSch\s*uld\s*un\s*fähig\s*keit', 'Schuldunfähigkeit'),
        (r'\bgesetz\s*buch', 'gesetzbuch'),
        (r'\bdefini\s*ert', 'definiert'),
        (r'\bgehand\s*elt', 'gehandelt'),
        (r'\bschuld\s*haft', 'schuldhaft'),
        (r'\bSt\s+ra\s*f\b', 'Straf'),
        (r'\bRe\s+chts\b', 'Rechts'),
        (r'\bW\s+ollen', 'Wollen'),
        (r'\bTä\s+ters?', lambda m: m.group().replace(' ', '')),
        (r'\bEins\s+icht', 'Einsicht'),
        (r'\bpers\s*önlich\w*', lambda m: m.group().replace(' ', '')),
        (r'\bAllgeme\s*ine?', lambda m: m.group().replace(' ', '')),
    ]
    for pattern, repl in fixes:
        if callable(repl):
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Rejoin single-letter fragments: "V ors atz" pattern
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


def _fix_markdown(text: str) -> str:
    """Fix malformed markdown patterns and OCR artifacts in LLM output."""
    # Fix bold markers with inner spaces: "** text **" → "**text**"
    text = re.sub(r"\*\*\s+(.+?)\s+\*\*", r"**\1**", text)
    # Fix bold markers wrapping words with inner spaces: "**V ors atz**" → "**Vorsatz**"
    def _fix_bold_content(m: re.Match) -> str:
        inner = m.group(1)
        # Rejoin broken words inside bold markers
        for _ in range(5):
            new = re.sub(r'([A-ZÄÖÜa-zäöüß]{1,})\s+([a-zäöüß]{1,4})\b', r'\1\2', inner)
            if new == inner:
                break
            inner = new
        return f"**{inner}**"
    text = re.sub(r"\*\*(.+?)\*\*", _fix_bold_content, text)
    # Fix numbered list items with extra spaces: "1 ." → "1."
    text = re.sub(r"(\d+)\s+\.", r"\1.", text)
    # Apply OCR word fixes to the full output
    text = _fix_ocr_words_in_output(text)
    return text


@dataclass
class RAGResult:
    answer: str
    sources: list[DocumentChunk]
    query_used: str


class RAGPipeline:
    """Core orchestrator: embed → retrieve → generate."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider

    async def _retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[list[DocumentChunk], list[float]]:
        """Embed the query and retrieve relevant chunks."""
        embedding = await embed_query(query)
        chunks = await search_documents(
            query_embedding=embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )
        return chunks, embedding

    def _build_messages(
        self,
        context: str,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the message array for the LLM."""
        settings = get_settings()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if context:
            messages.append({"role": "user", "content": context})
            messages.append(
                {
                    "role": "assistant",
                    "content": "Ich habe die Quellen gelesen und werde meine Antwort darauf stützen.",
                }
            )

        if conversation_history:
            turns = conversation_history[-settings.max_conversation_turns :]
            for msg in turns:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": query})
        return messages

    async def run(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> RAGResult:
        """Full RAG pipeline: embed → retrieve → generate (non-streaming)."""
        logger.info("rag_pipeline_started", query=query[:100])

        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        context = assemble_context(chunks)
        messages = self._build_messages(context, query, conversation_history)

        raw_answer = await self._llm.generate(messages)
        answer = _fix_markdown(raw_answer)

        logger.info("rag_pipeline_completed", sources=len(chunks))
        return RAGResult(answer=answer, sources=chunks, query_used=query)

    async def run_stream(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[AsyncGenerator[str, None], list[DocumentChunk]]:
        """
        Streaming RAG pipeline. Returns:
          - An async generator of answer tokens
          - The list of source chunks (available immediately after retrieval)
        """
        logger.info("rag_stream_started", query=query[:100])

        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        context = assemble_context(chunks)
        messages = self._build_messages(context, query, conversation_history)

        token_stream = self._llm.generate_stream(messages)
        return token_stream, chunks

    async def search_only(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> list[DocumentChunk]:
        """Vector search without LLM generation (for debugging retrieval)."""
        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        return chunks
