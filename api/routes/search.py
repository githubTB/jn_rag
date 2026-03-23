"""
api/routes/search.py — 搜索 + 问答接口。

接口
----
GET /api/search   纯向量检索，支持按企业/文件类型过滤
GET /api/query    向量检索 + Reranker 精排 + LLM 生成答案
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from config.settings import settings
from core.dedup import Dedup, DocType
from core.embedder import Embedder
from core.reranker import Reranker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


# ---------------------------------------------------------------------------
# 构建 Milvus filter 表达式
# ---------------------------------------------------------------------------

def _build_filter(
    company_id: str | None,
    doc_type: str | None,
    file_id: str | None,
) -> str | None:
    parts = []
    if company_id:
        parts.append(f'company_id == "{company_id}"')
    if doc_type:
        parts.append(f'doc_type == "{doc_type}"')
    if file_id:
        parts.append(f'file_id == "{file_id}"')
    return " and ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# GET /api/search — 纯向量检索（不经过 Reranker，调试用）
# ---------------------------------------------------------------------------

@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(5, ge=1, le=20, description="返回条数"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="最低相似度"),
    company_id: str | None = Query(None, description="限定企业范围"),
    doc_type: str | None = Query(None, description="限定文件类型"),
    file_id: str | None = Query(None, description="限定单个文件"),
):
    """
    纯向量检索，不经过 Reranker 和 LLM，适合调试和前端展示相关段落。
    """
    logger.info("[Search] q=%r company=%s type=%s", q, company_id, doc_type)

    if doc_type and doc_type not in DocType.ALL:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 doc_type: {doc_type}，可选: {', '.join(sorted(DocType.ALL))}",
        )

    filter_expr = _build_filter(company_id, doc_type, file_id)

    try:
        hits = Embedder.search(
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_expr=filter_expr,
        )
    except Exception as exc:
        logger.error("[Search] 检索失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}")

    return JSONResponse({
        "query":      q,
        "company_id": company_id,
        "doc_type":   doc_type,
        "total":      len(hits),
        "results":    hits,
    })


# ---------------------------------------------------------------------------
# GET /api/query — 向量检索 + Reranker 精排 + LLM 生成答案
# ---------------------------------------------------------------------------

# 向量粗筛倍数：top_k * RECALL_MULTIPLIER 条送入 Reranker
_RECALL_MULTIPLIER = 3
# 向量粗筛阈值（比 /search 放宽，让 Reranker 有更多候选）
_RECALL_THRESHOLD  = 0.2

@router.get("/query")
async def query(
    q: str = Query(..., min_length=1, description="问题"),
    top_k: int = Query(5, ge=1, le=20, description="最终返回条数（Reranker 精排后）"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="向量检索最低相似度"),
    company_id: str | None = Query(None, description="限定企业范围"),
    doc_type: str | None = Query(None, description="限定文件类型"),
    file_id: str | None = Query(None, description="限定单个文件"),
):
    """
    向量检索 + Reranker 精排 + Qwen 生成带来源标注的答案。

    流程：
        1. 向量粗筛：top_k × 3 条候选（阈值放宽到 0.2）
        2. Reranker 精排：Cross-encoder 打分，保留 top_k 条
        3. LLM 生成：基于精排结果构建 context，调用 Qwen 生成答案

    典型用法：
        ?q=注册资本是多少&company_id=company_001
        ?q=2024年发票总金额&company_id=company_001&doc_type=invoice
    """
    logger.info("[Query] q=%r company=%s type=%s", q, company_id, doc_type)

    if doc_type and doc_type not in DocType.ALL:
        raise HTTPException(status_code=400, detail=f"无效的 doc_type: {doc_type}")

    # ── STEP 1: 向量粗筛（多召回，给 Reranker 足够候选）──────────────
    filter_expr  = _build_filter(company_id, doc_type, file_id)
    recall_top_k = top_k * _RECALL_MULTIPLIER
    # score_threshold 用接口传入值和内部放宽值的较小值，保证召回量
    recall_threshold = min(score_threshold, _RECALL_THRESHOLD)

    try:
        hits = Embedder.search(
            query=q,
            top_k=recall_top_k,
            score_threshold=recall_threshold,
            filter_expr=filter_expr,
        )
    except Exception as exc:
        logger.error("[Query] 检索失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}")

    if not hits:
        return JSONResponse({
            "query":      q,
            "company_id": company_id,
            "answer":     "未找到相关内容，请确认文件已入库或放宽过滤条件。",
            "sources":    [],
            "chunks":     [],
            "reranker":   False,
        })

    logger.info("[Query] 向量粗筛: %d 条候选", len(hits))

    # ── STEP 2: Reranker 精排 ────────────────────────────────────────
    reranker_used = Reranker.is_available()
    hits = Reranker.rerank(query=q, hits=hits, top_n=top_k)
    logger.info("[Query] 精排后: %d 条", len(hits))

    # ── STEP 3: 拼装 context ─────────────────────────────────────────
    context = _build_context(hits)
    logger.debug("[Query] context 长度: %d 字符", len(context))

    # ── STEP 4: 获取企业名称 ─────────────────────────────────────────
    company_name: str | None = None
    if company_id:
        company = Dedup.get_company(company_id)
        if company:
            company_name = company["name"]

    # ── STEP 5: 调 LLM ───────────────────────────────────────────────
    answer = await _call_llm(q, context, company_name=company_name)

    # ── 来源汇总（去重，按文件归组）──────────────────────────────────
    seen: set[str] = set()
    sources = []
    for hit in hits:
        src = hit.get("source", "")
        if src not in seen:
            seen.add(src)
            sources.append({
                "file":          src.split("/")[-1],
                "path":          src,
                "doc_type":      hit.get("doc_type"),
                "company_id":    hit.get("company_id"),
                "score":         hit.get("score"),
                "rerank_score":  hit.get("rerank_score"),
            })

    logger.info("[Query] 答案 %d 字符，来源 %d 个，reranker=%s",
                len(answer), len(sources), reranker_used)

    return JSONResponse({
        "query":      q,
        "company_id": company_id,
        "answer":     answer,
        "sources":    sources,
        "chunks":     hits,
        "reranker":   reranker_used,
    })


# ---------------------------------------------------------------------------
# context 构建（按 doc_type 分组，保留 rerank_score）
# ---------------------------------------------------------------------------

_DOC_TYPE_LABEL: dict[str, str] = {
    DocType.LICENSE:   "营业执照",
    DocType.INVOICE:   "发票/单据",
    DocType.TABLE:     "表格/报表",
    DocType.NAMEPLATE: "设备铭牌",
    DocType.DOCUMENT:  "文档",
    DocType.UNKNOWN:   "其他",
}


# 替换 search.py 里的 _build_context 函数

def _build_context(hits: list[dict]) -> str:
    """
    将精排后的结果按 doc_type 分组拼装 context。

    表格类型（label=table 或 doc_type=table）优先使用 raw_content
    保留原始格式（HTML/Markdown），让 LLM 能看到完整表格结构。
    其他类型使用 content（纯文本）。
    """
    groups: dict[str, list[dict]] = {}
    for hit in hits:
        dt = hit.get("doc_type") or DocType.UNKNOWN
        groups.setdefault(dt, []).append(hit)

    parts = []
    ref_idx = 1
    for dt, group_hits in groups.items():
        label = _DOC_TYPE_LABEL.get(dt, dt)
        parts.append(f"【{label}】")
        for hit in group_hits:
            source_name = (hit.get("source") or "").split("/")[-1]
            score_info  = (
                f"相关度 {hit['rerank_score']:.2f}"
                if "rerank_score" in hit
                else f"相似度 {hit.get('score', 0):.2f}"
            )

            # 表格类型用 raw_content 保留格式，其他用 content 纯文本
            is_table = (
                dt == DocType.TABLE
                or hit.get("label") in ("table", "ocr")
                and hit.get("raw_content", "") != hit.get("content", "")
            )
            body = hit.get("raw_content") or hit.get("content", "") if is_table \
                else hit.get("content", "")

            parts.append(
                f"[{ref_idx}] 来源：{source_name}（{score_info}）\n{body}"
            )
            ref_idx += 1

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

async def _call_llm(
    question: str,
    context: str,
    company_name: str | None = None,
) -> str:
    """调用 Qwen 生成答案，LLM 不可用时降级返回原始检索内容。"""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    client = AsyncOpenAI(
        base_url=settings.llm_api_base or "http://localhost:8000/v1",
        api_key=settings.llm_api_key or "EMPTY",
    )

    company_context = f"当前分析的企业：{company_name}\n" if company_name else ""

    system_prompt = (
        f"{company_context}"
        "你是一个专业的企业文档分析助手。"
        "请根据提供的文档内容准确回答问题，答案简洁专业。"
        "文档内容已按类型分组（营业执照、发票、表格等），请结合各类型内容综合分析。"
        "引用来源时用 [序号] 标注，例如：根据营业执照 [1] 显示..."
        "如果文档内容不足以回答，请明确说明缺少哪类文件。"
    )

    user_prompt = f"文档内容：\n{context}\n\n问题：{question}"

    logger.debug("[LLM] model=%s company=%s", settings.llm_model, company_name)

    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = resp.choices[0].message.content or ""
        logger.debug("[LLM] 完成，tokens=%s", resp.usage)
        return answer.strip()

    except Exception as exc:
        logger.error("[LLM] 调用失败: %s", exc, exc_info=True)
        return f"（LLM 暂不可用，以下为原始检索内容）\n\n{context[:1500]}"
        