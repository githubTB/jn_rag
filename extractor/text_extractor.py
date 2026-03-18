from pathlib import Path

from .base import BaseExtractor
from .helpers import detect_file_encodings
from models.document import Document


class TextExtractor(BaseExtractor):
    def __init__(self, file_path: str, encoding: str | None = None, autodetect_encoding: bool = True):
        self._file_path = file_path
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding

    def extract(self) -> list[Document]:
        try:
            text = Path(self._file_path).read_text(encoding=self._encoding)
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                for enc in detect_file_encodings(self._file_path):
                    try:
                        text = Path(self._file_path).read_text(encoding=enc.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(f"All detected encodings failed for {self._file_path}") from e
            else:
                raise RuntimeError(f"Decode failed for {self._file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self._file_path}") from e

        return [Document(page_content=text, metadata={"source": self._file_path})]
