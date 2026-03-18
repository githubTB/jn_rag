import csv

import pandas as pd

from .base import BaseExtractor
from .helpers import detect_file_encodings
from models.document import Document


class CSVExtractor(BaseExtractor):
    def __init__(
        self,
        file_path: str,
        encoding: str | None = None,
        autodetect_encoding: bool = True,
        source_column: str | None = None,
        csv_args: dict | None = None,
    ):
        self._file_path = file_path
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding
        self.source_column = source_column
        self.csv_args = csv_args or {}

    def extract(self) -> list[Document]:
        try:
            with open(self._file_path, newline="", encoding=self._encoding) as f:
                return self._read(f)
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                for enc in detect_file_encodings(self._file_path):
                    try:
                        with open(self._file_path, newline="", encoding=enc.encoding) as f:
                            return self._read(f)
                    except UnicodeDecodeError:
                        continue
            raise RuntimeError(f"Error loading {self._file_path}") from e

    def _read(self, csvfile) -> list[Document]:
        docs = []
        try:
            df = pd.read_csv(csvfile, on_bad_lines="skip", **self.csv_args)
            if self.source_column and self.source_column not in df.columns:
                raise ValueError(f"Source column '{self.source_column}' not found.")
            for i, row in df.iterrows():
                content = "; ".join(f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns)
                source = str(row[self.source_column]) if self.source_column else self._file_path
                docs.append(Document(page_content=content, metadata={"source": source, "row": i}))
        except csv.Error as e:
            raise RuntimeError(f"CSV parse error: {e}") from e
        return docs
