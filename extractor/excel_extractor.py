import os
import re
from zipfile import BadZipFile
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
        detected = self._detect_excel_format()
        
        if detected == "xlsx":
            try:
                wb = load_workbook(self._file_path, read_only=True, data_only=True)
            except BadZipFile as exc:
                raise ValueError(
                    f"Excel 文件扩展名为 {ext}，但文件内容不是合法的 xlsx 压缩包: "
                    f"{os.path.basename(self._file_path)}"
                ) from exc
            try:
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    header_row_idx, col_map, max_col = self._find_header(sheet)
                    if not col_map:
                        continue
                    for row in sheet.iter_rows(min_row=header_row_idx + 1, max_col=max_col, values_only=False):
                        if all(cell.value is None for cell in row):
                            continue
                        row_values: dict[int, str] = {}
                        for col_idx, cell in enumerate(row):
                            if col_idx not in col_map:
                                continue
                            raw = cell.value
                            row_values[col_idx] = "" if raw is None else str(raw).strip()

                        if self._is_repeated_header_row(row_values, col_map):
                            continue
                        if not self._has_non_key_cell_value(row_values, col_map):
                            # 通用且保守：多列表格中，除首列外全空的行通常是标题/分隔行
                            continue
                        if self._is_sparse_auxiliary_row(row_values, col_map):
                            # 通用且保守：首列为空、仅一个辅助列有值的行通常是单位/注释行
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

        elif detected == "xls":
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
        elif detected == "html":
            tables = pd.read_html(self._file_path)
            for idx, df in enumerate(tables, start=1):
                df.dropna(how="all", inplace=True)
                for _, row in df.iterrows():
                    parts = [f'"{k}":"{v}"' for k, v in row.items() if pd.notna(v)]
                    if parts:
                        documents.append(
                            Document(
                                page_content="; ".join(parts),
                                metadata={"source": self._file_path, "sheet": f"table_{idx}"},
                            )
                        )
        else:
            raise ValueError(f"不支持的 Excel 文件格式: ext={ext}, file={os.path.basename(self._file_path)}。"
    " 文件可能并不是真正的 Excel，或扩展名与实际内容不一致。")

        return documents

    def _detect_excel_format(self) -> str | None:
        with open(self._file_path, "rb") as f:
            magic = f.read(512)

        if magic.startswith(b"PK\x03\x04"):
            return "xlsx"
        if magic.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
            return "xls"
            
        head = magic.lstrip().lower()
        if head.startswith(b"<html") or head.startswith(b"<!doctype html"):
            return "html"

        ext = os.path.splitext(self._file_path)[-1].lower()
        if ext == ".xlsx":
            return "xlsx"
        if ext == ".xls":
            return "xls"
        return None

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

    def _is_repeated_header_row(self, row_values: dict[int, str], col_map: dict[int, str]) -> bool:
        """
        过滤重复表头行：
        例如在中途再次出现 “品名 / 2021年 / 2022年 / 2023年”。
        """
        non_empty = {k: v for k, v in row_values.items() if v}
        if not non_empty:
            return False
        matched = 0
        for col_idx, value in non_empty.items():
            if self._normalize_text(value) == self._normalize_text(col_map.get(col_idx, "")):
                matched += 1
        return matched >= max(2, len(non_empty) - 1)

    def _has_non_key_cell_value(self, row_values: dict[int, str], col_map: dict[int, str]) -> bool:
        """
        通用数据有效性判断：
        - 若表头只有 1 列，只要该列非空即可
        - 若表头有多列，要求至少一个“非首列”有值
        """
        if not col_map:
            return False
        ordered_cols = sorted(col_map.keys())
        if len(ordered_cols) == 1:
            return bool((row_values.get(ordered_cols[0]) or "").strip())
        for col_idx in ordered_cols[1:]:
            if (row_values.get(col_idx) or "").strip():
                return True
        return False

    def _is_sparse_auxiliary_row(self, row_values: dict[int, str], col_map: dict[int, str]) -> bool:
        if not col_map:
            return False
        ordered_cols = sorted(col_map.keys())
        if len(ordered_cols) < 2:
            return False

        first_col = ordered_cols[0]
        first_val = (row_values.get(first_col) or "").strip()
        if first_val:
            return False

        non_key_non_empty = [
            (c, (row_values.get(c) or "").strip())
            for c in ordered_cols[1:]
            if (row_values.get(c) or "").strip()
        ]
        return len(non_key_non_empty) == 1

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", "", value).strip().lower()
