from abc import ABC, abstractmethod

from models.document import Document


class BaseExtractor(ABC):
    """Abstract base class for all document extractors."""

    @abstractmethod
    def extract(self) -> list[Document]:
        raise NotImplementedError
