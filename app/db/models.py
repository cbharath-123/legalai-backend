from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class DocumentChunk:
    id: UUID
    content: str
    metadata: dict = field(default_factory=dict)
    similarity: float = 0.0

    @property
    def source_display(self) -> str:
        """Human-readable source label from metadata."""
        title = self.metadata.get("title", "")
        section = self.metadata.get("section", "")
        if title and section:
            return f"{title} – {section}"
        return title or section or f"Dokument #{self.id}"


@dataclass
class ConversationMessage:
    role: str  # "user" | "assistant"
    content: str
