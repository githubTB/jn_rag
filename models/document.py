from dataclasses import dataclass, field


@dataclass
class Document:
    """A piece of extracted content with optional metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", " ")
        return f"Document(content={preview!r}, metadata={self.metadata})"