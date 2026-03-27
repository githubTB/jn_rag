from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.dedup import DocType
from models.document import Document


@dataclass(frozen=True)
class DocTypeDecision:
    doc_type: str
    confidence: float
    evidence: list[str]


_KEYWORDS: dict[str, tuple[str, ...]] = {
    DocType.LICENSE: (
        "营业执照",
        "统一社会信用代码",
        "法定代表人",
        "注册资本",
        "成立日期",
        "公司类型",
    ),
    DocType.INVOICE: (
        "发票",
        "税率",
        "价税合计",
        "开票日期",
        "购买方",
        "销售方",
        "纳税人识别号",
    ),
    DocType.NAMEPLATE: (
        "型号",
        "规格",
        "额定功率",
        "额定电压",
        "制造商",
        "出厂编号",
        "产品编号",
        "serial",
        "model",
    ),
}


def classify_doc_type(
    docs: list[Document],
    file_path: str | Path | None = None,
) -> DocTypeDecision:
    if not docs:
        return DocTypeDecision(DocType.UNKNOWN, 0.0, ["no_docs"])

    path = Path(file_path) if file_path is not None else None
    suffix = path.suffix.lower() if path else ""
    labels = [str(doc.metadata.get("label", "")).lower() for doc in docs]
    all_text = "\n".join(doc.page_content for doc in docs if doc.page_content).lower()

    if suffix in {".xlsx", ".xls", ".csv"}:
        return DocTypeDecision(DocType.TABLE, 0.99, [f"suffix:{suffix}"])

    if not all_text.strip() and suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return DocTypeDecision(DocType.UNKNOWN, 0.1, ["empty_ocr"])

    scores: dict[str, float] = {
        DocType.LICENSE: 0.0,
        DocType.INVOICE: 0.0,
        DocType.TABLE: 0.0,
        DocType.NAMEPLATE: 0.0,
        DocType.DOCUMENT: 0.2,
    }
    evidence: dict[str, list[str]] = {key: [] for key in scores}

    table_blocks = labels.count("table")
    ocr_blocks = labels.count("ocr")
    seal_blocks = labels.count("seal")
    title_blocks = labels.count("title")
    text_blocks = labels.count("text")
    if docs and table_blocks / len(docs) >= 0.6:
        scores[DocType.TABLE] += 0.75
        evidence[DocType.TABLE].append(f"table_blocks:{table_blocks}/{len(docs)}")
    elif table_blocks > 0:
        scores[DocType.TABLE] += 0.35
        evidence[DocType.TABLE].append(f"table_blocks:{table_blocks}")

    if ocr_blocks == len(docs) and docs:
        scores[DocType.NAMEPLATE] += 0.1
        evidence[DocType.NAMEPLATE].append("ocr_only_blocks")

    if seal_blocks > 0 and title_blocks > 0:
        scores[DocType.LICENSE] += 0.25
        evidence[DocType.LICENSE].append("structure:seal+title")

    if table_blocks > 0 and text_blocks > 0:
        scores[DocType.INVOICE] += 0.15
        evidence[DocType.INVOICE].append("structure:table+text")

    if ocr_blocks > 0 and len(docs) <= 3 and table_blocks == 0:
        scores[DocType.NAMEPLATE] += 0.2
        evidence[DocType.NAMEPLATE].append("structure:compact_ocr")

    for doc_type, keywords in _KEYWORDS.items():
        hits = [kw for kw in keywords if kw.lower() in all_text]
        if not hits:
            continue
        scores[doc_type] += min(0.85, 0.22 * len(hits))
        evidence[doc_type].append("keywords:" + ",".join(hits[:4]))

    if "金额" in all_text and "税额" in all_text:
        scores[DocType.INVOICE] += 0.2
        evidence[DocType.INVOICE].append("invoice_fields")

    if "设备" in all_text and ("型号" in all_text or "功率" in all_text):
        scores[DocType.NAMEPLATE] += 0.2
        evidence[DocType.NAMEPLATE].append("equipment_fields")

    if "单位" in all_text and "数值" in all_text:
        scores[DocType.TABLE] += 0.15
        evidence[DocType.TABLE].append("tabular_fields")

    winner = max(scores, key=scores.get)
    confidence = max(0.0, min(scores[winner], 0.99))

    if confidence < 0.45:
        fallback = DocType.UNKNOWN if suffix in {
            ".pdf", ".docx", ".docm", ".pptx",
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
        } else DocType.DOCUMENT
        return DocTypeDecision(fallback, confidence, evidence.get(winner) or ["low_confidence"])

    return DocTypeDecision(winner, confidence, evidence[winner] or ["content_match"])
