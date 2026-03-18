import os
from typing import TypedDict

import pandas as pd
from openpyxl import load_workbook

from .base import BaseExtractor
from models.document import Document


class _Candidate(TypedDict):
    idx: int
    count: int
    map: dict[int, str]


class ExcelExtractor(BaseExtractor):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def extract(self) -> list[Document]:
        documents: list[Document] = []
        ext = os.path.splitext(self._file_path)[-1].lower()

        if ext == ".xlsx":
            wb = load_workbook(self._file_path, read_only=True, data_only=True)
            try:
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    header_row_idx, col_map, max_col = self._find_header(sheet)
                    if not col_map:
                        continue
                    for row in sheet.iter_rows(min_row=header_row_idx + 1, max_col=max_col, values_only=False):
                        if all(cell.value is None for cell in row):
                            continue
                        parts = []
                        for col_idx, cell in enumerate(row):
                            if col_idx not in col_map:
                                continue
                            col_name = col_map[col_idx]
                            value = cell.value
                            if hasattr(cell, "hyperlink") and cell.hyperlink:
                                target = getattr(cell.hyperlink, "target", None)
                                if target:
                                    value = f"[{value}]({target})"
                            value = "" if value is None else str(value).strip().replace('"', '\\"')
                            parts.append(f'"{col_name}":"{value}"')
                        if parts:
                            documents.append(
                                Document(
                                    page_content="; ".join(parts),
                                    metadata={"source": self._file_path, "sheet": sheet_name},
                                )
                            )
            finally:
                wb.close()

        elif ext == ".xls":
            excel_file = pd.ExcelFile(self._file_path, engine="xlrd")
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name=sheet_name)
                df.dropna(how="all", inplace=True)
                for _, row in df.iterrows():
                    parts = [f'"{k}":"{v}"' for k, v in row.items() if pd.notna(v)]
                    documents.append(
                        Document(
                            page_content="; ".join(parts),
                            metadata={"source": self._file_path, "sheet": sheet_name},
                        )
                    )
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return documents

    def _find_header(self, sheet, scan_rows: int = 10) -> tuple[int, dict[int, str], int]:
        candidates: list[_Candidate] = []
        for row_idx, row in enumerate(sheet.iter_rows(min_row=1, max_row=scan_rows, values_only=True), start=1):
            row_map: dict[int, str] = {}
            for col_idx, val in enumerate(row):
                if val is not None and str(val).strip():
                    row_map[col_idx] = str(val).strip().replace('"', '\\"')
            if not row_map:
                continue
            candidates.append({"idx": row_idx, "count": len(row_map), "map": row_map})

        if not candidates:
            return 0, {}, 0

        best: _Candidate | None = None
        for c in candidates:
            if c["count"] >= 2:
                best = c
                break
        if not best:
            candidates.sort(key=lambda x: (-x["count"], x["idx"]))
            best = candidates[0]

        max_col = max(best["map"].keys()) + 1
        return best["idx"], best["map"], max_col
