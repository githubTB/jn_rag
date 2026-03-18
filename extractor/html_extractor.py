from bs4 import BeautifulSoup

from .base import BaseExtractor
from models.document import Document


class HtmlExtractor(BaseExtractor):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def extract(self) -> list[Document]:
        with open(self._file_path, "rb") as fp:
            soup = BeautifulSoup(fp, "html.parser")
            text = soup.get_text(separator="\n").strip()
        return [Document(page_content=text, metadata={"source": self._file_path})]
