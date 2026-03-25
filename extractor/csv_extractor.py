import csv

import pandas as pd

from .base import BaseExtractor
from .helpers import detect_file_encodings
from models.document import Document


class CSVExtractor(BaseExtractor):
    WINDOW_ROWS = 40

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
            columns = [str(col).strip() for col in df.columns if str(col).strip()]
            row_records: list[dict[str, object]] = []
            for i, row in df.iterrows():
                row_map: dict[str, str] = {}
                for col in columns:
                    raw = row[col]
                    value = "" if pd.isna(raw) else str(raw).strip()
                    row_map[col] = value
                if not any(v.strip() for v in row_map.values()):
                    continue
                row_number = int(i) + 1
                row_records.append({"row_number": row_number, "row_map": row_map})
                content = self._format_row_content(row_map, row_number)
                source = str(row[self.source_column]) if self.source_column else self._file_path
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "row": i,
                            "label": "table",
                            "granularity": "row",
                            "row_number": row_number,
                        },
                    )
                )
            docs.extend(self._build_aggregate_documents(columns, row_records))
        except csv.Error as e:
            raise RuntimeError(f"CSV parse error: {e}") from e
        return docs

    @staticmethod
    def _escape_text(value: str) -> str:
        return value.replace('"', '\\"')

    def _format_row_content(self, row_map: dict[str, str], row_number: int) -> str:
        parts = [
            f'"文件":"{self._escape_text(self._file_path)}"',
            f'"行号":"{row_number}"',
        ]
        for col, value in row_map.items():
            parts.append(f'"{col}":"{self._escape_text(value)}"')
        return "; ".join(parts)

    def _build_aggregate_documents(
        self,
        columns: list[str],
        row_records: list[dict[str, object]],
    ) -> list[Document]:
        if not row_records:
            return []

        docs: list[Document] = []
        source = self._file_path
        total = len(row_records)
        for start in range(0, total, self.WINDOW_ROWS):
            window_rows = row_records[start:start + self.WINDOW_ROWS]
            if len(window_rows) <= 1:
                continue
            docs.append(
                Document(
                    page_content=self._format_table_block(columns, window_rows, "window"),
                    metadata={
                        "source": source,
                        "label": "table",
                        "granularity": "window",
                        "row_start": window_rows[0]["row_number"],
                        "row_end": window_rows[-1]["row_number"],
                    },
                )
            )

        if total > 1:
            docs.append(
                Document(
                    page_content=self._format_table_block(columns, row_records, "file"),
                    metadata={
                        "source": source,
                        "label": "table",
                        "granularity": "sheet",
                        "row_start": row_records[0]["row_number"],
                        "row_end": row_records[-1]["row_number"],
                    },
                )
            )
        return docs

    @staticmethod
    def _format_table_block(
        columns: list[str],
        row_records: list[dict[str, object]],
        granularity: str,
    ) -> str:
        lines = [
            "工作表：CSV",
            f"粒度：{granularity}",
            "表头：" + "\t".join(columns),
        ]
        for record in row_records:
            row_map = record.get("row_map", {})
            row_number = record.get("row_number")
            values = [str(row_map.get(col, "")).replace("\n", " ").strip() for col in columns]
            lines.append(f"第{row_number}行\t" + "\t".join(values))
        return "\n".join(lines)
