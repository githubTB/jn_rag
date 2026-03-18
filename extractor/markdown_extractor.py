import re
from pathlib import Path

from .base import BaseExtractor
from .helpers import detect_file_encodings
from models.document import Document


class MarkdownExtractor(BaseExtractor):
    def __init__(
        self,
        file_path: str,
        remove_hyperlinks: bool = False,
        remove_images: bool = False,
        encoding: str | None = None,
        autodetect_encoding: bool = True,
    ):
        self._file_path = file_path
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding

    def extract(self) -> list[Document]:
        tups = self._parse_tups(self._file_path)
        documents = []
        for header, value in tups:
            value = value.strip()
            if not value:
                continue
            if header is None:
                documents.append(Document(page_content=value, metadata={"source": self._file_path}))
            else:
                documents.append(
                    Document(
                        page_content=f"{header}\n{value}",
                        metadata={"source": self._file_path, "header": header},
                    )
                )
        return documents

    def _markdown_to_tups(self, text: str) -> list[tuple[str | None, str]]:
        tups: list[tuple[str | None, str]] = []
        lines = text.split("\n")
        current_header: str | None = None
        current_text = ""
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                current_text += line + "\n"
                continue
            if in_code_block:
                current_text += line + "\n"
                continue
            if re.match(r"^#+\s", line):
                tups.append((current_header, current_text))
                current_header = re.sub(r"^#+\s*", "", line).strip()
                current_text = ""
            else:
                current_text += line + "\n"
        tups.append((current_header, current_text))
        return tups

    def _parse_tups(self, filepath: str) -> list[tuple[str | None, str]]:
        content = ""
        try:
            content = Path(filepath).read_text(encoding=self._encoding)
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                for enc in detect_file_encodings(filepath):
                    try:
                        content = Path(filepath).read_text(encoding=enc.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {filepath}") from e

        if self._remove_hyperlinks:
            content = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", content)
        if self._remove_images:
            content = re.sub(r"!{1}\[\[(.*)\]\]", "", content)

        return self._markdown_to_tups(content)
