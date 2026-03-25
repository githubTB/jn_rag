"""
api/routes/search.py — 搜索 + 问答接口。

接口
----
GET /api/search   纯向量检索，支持按企业/文件类型过滤
GET /api/query    向量检索 + Reranker 精排 + LLM 生成答案
"""

from __future__ import annotations

import json
import logging
from math import log
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from services.mcp_service import (
    BASE_EXTRACT_MCP_SERVICE,
    get_service,
    list_service_names,
)
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

_DEFAULT_EXTRACT_PROMPT = get_service(BASE_EXTRACT_MCP_SERVICE)
_DEFAULT_EXTRACT_Q = _DEFAULT_EXTRACT_PROMPT.q
_DEFAULT_EXTRACT_SYSTEM_PROMPT = _DEFAULT_EXTRACT_PROMPT.system_prompt
_DEFAULT_EXTRACT_USER_TEMPLATE = _DEFAULT_EXTRACT_PROMPT.user_prompt


def _resolve_with_service_default(
    incoming: str | None,
    *,
    base_default: str,
    service_default: str,
) -> str:
    """
    解析字段最终值：
    - 未传/空字符串 -> 使用 service 默认值
    - 传入值等于 base 默认值 -> 视为前端占位默认值，使用 service 默认值
    - 其他情况 -> 使用传入值
    """
    if incoming is None:
        return service_default
    normalized = incoming.strip()
    if not normalized:
        return service_default
    if normalized == base_default:
        return service_default
    return incoming


def _preview_for_log(text: str, limit: int = 1200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_matched_sources(hits: list[dict]) -> None:
    """
    打印向量检索匹配到的源文档内容。
    重点格式：file_id:源文档内容
    """
    if not hits:
        logger.info("[RetrieveHit] 无命中")
        return

    for idx, hit in enumerate(hits, start=1):
        file_id = str(hit.get("file_id") or "unknown_file_id")
        source = str(hit.get("source") or "")
        content = str(hit.get("raw_content") or hit.get("content") or "")
        logger.info(
            "[RetrieveHit] #%s %s:%s source=%s",
            idx,
            file_id,
            _preview_for_log(content),
            source,
        )


class QueryExtractRequest(BaseModel):
    service_name: str = Field(BASE_EXTRACT_MCP_SERVICE, description="模板服务名")
    q: str | None = Field(None, min_length=1, description="检索问题/抽取主题（不传则使用服务默认值）")
    top_k: int = Field(5, ge=1, le=20, description="最终保留条数")
    score_threshold: float = Field(0.3, ge=0.0, le=1.0, description="向量检索最低相似度")
    company_id: str | None = Field(None, description="限定企业范围")
    doc_type: str | None = Field(None, description="限定文件类型")
    file_id: str | None = Field(None, description="限定单个文件")
    prefer_source: str | None = Field(None, description="优先命中的源文件名关键词")
    system_prompt: str | None = Field(None, description="系统提示词（不传则使用服务默认值）")
    user_prompt_template: str | None = Field(
        None,
        description="用户提示词模板，必须包含 {text} 占位符（不传则使用服务默认值）",
    )
    max_tokens: int = Field(2000, ge=1, le=8192, description="LLM 最大输出 token")
    include_chunks: bool = Field(True, description="是否返回召回片段")
    include_raw_llm_output: bool = Field(False, description="是否返回 LLM 原始文本")


def _retrieve_hits(
    *,
    q: str,
    top_k: int,
    score_threshold: float,
    company_id: str | None,
    doc_type: str | None,
    file_id: str | None,
    prefer_source: str | None = None,
) -> tuple[list[dict], bool]:
    if doc_type and doc_type not in DocType.ALL:
        raise HTTPException(status_code=400, detail=f"无效的 doc_type: {doc_type}")

    filter_expr = _build_filter(company_id, doc_type, file_id)
    recall_top_k = top_k * _RECALL_MULTIPLIER
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
        return [], False

    logger.info("[Query] 向量粗筛: %d 条候选", len(hits))
    reranker_used = Reranker.is_available()
    rerank_top_n = len(hits) if prefer_source else top_k
    hits = Reranker.rerank(query=q, hits=hits, top_n=rerank_top_n)
    if prefer_source:
        keyword = prefer_source.strip().lower()
        if keyword:
            hits.sort(
                key=lambda h: (
                    keyword in ((h.get("source") or "").split("/")[-1].lower()),
                    h.get("rerank_score", h.get("score", 0)),
                ),
                reverse=True,
            )
            logger.info("[Query] 已应用 prefer_source 加权: %r", prefer_source)
        hits = hits[:top_k]
    logger.info("[Query] 精排后: %d 条", len(hits))
    _log_matched_sources(hits)
    return hits, reranker_used

@router.get("/query")
async def query(
    q: str = Query(..., min_length=1, description="问题"),
    top_k: int = Query(5, ge=1, le=20, description="最终返回条数（Reranker 精排后）"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="向量检索最低相似度"),
    company_id: str | None = Query(None, description="限定企业范围"),
    doc_type: str | None = Query(None, description="限定文件类型"),
    file_id: str | None = Query(None, description="限定单个文件"),
    prefer_source: str | None = Query(None, description="优先命中的源文件名关键词"),
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
    hits, reranker_used = _retrieve_hits(
        q=q,
        top_k=top_k,
        score_threshold=score_threshold,
        company_id=company_id,
        doc_type=doc_type,
        file_id=file_id,
        prefer_source=prefer_source,
    )

    if not hits:
        return JSONResponse({
            "query":      q,
            "company_id": company_id,
            "answer":     "未找到相关内容，请确认文件已入库或放宽过滤条件。",
            "sources":    [],
            "chunks":     [],
            "reranker":   False,
        })

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


@router.post("/query/extract")
async def query_extract(body: QueryExtractRequest):
    """
    检索 + Reranker + LLM 结构化提取。

    适合把企业文档中的关键字段抽取为 JSON。
    """
    logger.info(
        "[QueryExtract] service=%s q=%r company=%s type=%s",
        body.service_name,
        body.q,
        body.company_id,
        body.doc_type,
    )

    if body.service_name not in list_service_names():
        raise HTTPException(
            status_code=400,
            detail=f"无效的 service_name: {body.service_name}，可选: {', '.join(list_service_names())}",
        )

    selected_service = get_service(body.service_name)
    logger.info("[QueryExtract] selected_service=%s", selected_service)
    
    request_q = _resolve_with_service_default(
        body.q,
        base_default=_DEFAULT_EXTRACT_Q,
        service_default=selected_service.q,
    )
    request_system_prompt = _resolve_with_service_default(
        body.system_prompt,
        base_default=_DEFAULT_EXTRACT_SYSTEM_PROMPT,
        service_default=selected_service.system_prompt,
    )
    request_user_prompt_template = _resolve_with_service_default(
        body.user_prompt_template,
        base_default=_DEFAULT_EXTRACT_USER_TEMPLATE,
        service_default=selected_service.user_prompt,
    )

    if "{text}" not in request_user_prompt_template:
        raise HTTPException(status_code=400, detail="user_prompt_template 必须包含 {text} 占位符")

    hits, reranker_used = _retrieve_hits(
        q=request_q,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        company_id=body.company_id,
        doc_type=body.doc_type,
        file_id=body.file_id,
        prefer_source=body.prefer_source,
    )

    if not hits:
        return JSONResponse({
            "query": request_q,
            "company_id": body.company_id,
            "structured_data": None,
            "raw_output": None,
            "sources": [],
            "chunks": [],
            "reranker": False,
            "message": "未找到相关内容，请确认文件已入库或放宽过滤条件。",
        })

    context = _build_context(hits)

    raw_output, structured_data = await _call_llm_extract(
        text=context,
        system_prompt=request_system_prompt,
        user_prompt_template=request_user_prompt_template,
        max_tokens=body.max_tokens,
    )

    seen: set[str] = set()
    sources = []
    for hit in hits:
        src = hit.get("source", "")
        if src not in seen:
            seen.add(src)
            sources.append({
                "file": src.split("/")[-1],
                "path": src,
                "doc_type": hit.get("doc_type"),
                "company_id": hit.get("company_id"),
                "score": hit.get("score"),
                "rerank_score": hit.get("rerank_score"),
            })

    response: dict = {
        "query": request_q,
        "service_name": body.service_name,
        "company_id": body.company_id,
        "structured_data": structured_data,
        "sources": sources,
        "reranker": reranker_used,
    }
    if body.include_chunks:
        response["chunks"] = hits
    if body.include_raw_llm_output:
        response["raw_output"] = raw_output
    return JSONResponse(response)


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


async def _call_llm_extract(
    *,
    text: str,
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int,
) -> tuple[str, dict | list | None]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    client = AsyncOpenAI(
        base_url=settings.llm_api_base or "http://localhost:8000/v1",
        api_key=settings.llm_api_key or "EMPTY",
    )
    user_prompt = user_prompt_template.format(text=text)

    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        logger.error("[LLMExtract] 调用失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM 提取失败: {exc}")

    raw = (resp.choices[0].message.content or "").strip()
    finish_reason = resp.choices[0].finish_reason
    logger.debug("[LLMExtract] finish_reason=%s 原始输出: %s", finish_reason, raw[:500])

    parsed = _safe_parse_json(raw)
    # 输出被 max_tokens 截断时，自动放大 token 重试一次，减少半截 JSON。
    if parsed is None and finish_reason == "length" and max_tokens < 8192:
        retry_max_tokens = min(8192, max_tokens * 2)
        logger.warning(
            "[LLMExtract] 输出疑似截断，重试一次 max_tokens=%s -> %s",
            max_tokens,
            retry_max_tokens,
        )
        try:
            retry_resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=retry_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            retry_raw = (retry_resp.choices[0].message.content or "").strip()
            retry_finish_reason = retry_resp.choices[0].finish_reason
            logger.debug(
                "[LLMExtract] retry finish_reason=%s 原始输出: %s",
                retry_finish_reason,
                retry_raw[:500],
            )
            retry_parsed = _safe_parse_json(retry_raw)
            if retry_parsed is not None:
                return retry_raw, retry_parsed
        except Exception as exc:
            logger.warning("[LLMExtract] 重试失败: %s", exc)

    return raw, parsed


def _safe_parse_json(text: str) -> dict | list | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]

    # 容忍模型在 JSON 前后加解释文本，尝试抽取第一个完整 JSON 块。
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, (dict, list)):
                return data
        except Exception:
            continue

    logger.warning("[LLMExtract] JSON 解析失败，保留原始输出")
    return None
        
