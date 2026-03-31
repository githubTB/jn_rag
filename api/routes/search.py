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
import time

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


def _llm_usage_payload(
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    duration_ms: int | None = None,
) -> dict:
    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "duration_ms": duration_ms,
    }


def _extract_usage(resp) -> tuple[int | None, int | None]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _public_hit(hit: dict) -> dict:
    public = dict(hit)
    public["task_id"] = public.get("task_id")
    return public


# ---------------------------------------------------------------------------
# 构建 Milvus filter 表达式
# ---------------------------------------------------------------------------

def _build_filter(
    task_id: str | None,
    doc_type: str | None,
    file_id: str | None,
) -> str | None:
    parts = []
    if task_id:
        parts.append(f'task_id == "{task_id}"')
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
    task_id: str | None = Query(None, description="限定任务范围"),
    doc_type: str | None = Query(None, description="限定文件类型"),
    file_id: str | None = Query(None, description="限定单个文件"),
):
    """
    纯向量检索，不经过 Reranker 和 LLM，适合调试和前端展示相关段落。
    """
    logger.info("[Search] q=%r task=%s type=%s", q, task_id, doc_type)

    if doc_type and doc_type not in DocType.ALL:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 doc_type: {doc_type}，可选: {', '.join(sorted(DocType.ALL))}",
        )

    filter_expr = _build_filter(task_id, doc_type, file_id)

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
        "task_id":    task_id,
        "doc_type":   doc_type,
        "total":      len(hits),
        "results":    [_public_hit(hit) for hit in hits],
    })


# ---------------------------------------------------------------------------
# GET /api/query — 向量检索 + Reranker 精排 + LLM 生成答案
# ---------------------------------------------------------------------------

# 向量粗筛倍数：top_k * RECALL_MULTIPLIER 条送入 Reranker
_RECALL_MULTIPLIER = 3
# 向量粗筛阈值（比 /search 放宽，让 Reranker 有更多候选）
_RECALL_THRESHOLD  = 0.2
_PARENT_MAX_PER_FILE = 2

_DEFAULT_EXTRACT_PROMPT = get_service(BASE_EXTRACT_MCP_SERVICE)
_DEFAULT_EXTRACT_Q = _DEFAULT_EXTRACT_PROMPT.q
_DEFAULT_EXTRACT_SYSTEM_PROMPT = _DEFAULT_EXTRACT_PROMPT.system_prompt
_DEFAULT_EXTRACT_USER_TEMPLATE = _DEFAULT_EXTRACT_PROMPT.user_prompt

_SERVICE_CN_NAME_BY_NAME: dict[str, str] = {
    "base_extract_prompt_template": "企业基本信息",
    "product_extract_prompt_template": "近三年产品产量",
    "business_scale_extract_prompt_template": "经营规模信息",
    "process_flow_extract_prompt_template": "生产工艺流程",
    "major_production_equipment_extract_prompt_template": "主要生产设备信息",
    "regulation_policy_extract_prompt_template": "相关法规政策依据清单",
    "transformer_efficiency_benchmark_extract_prompt_template": "变压器设备能效对标",
    "air_compressor_efficiency_benchmark_extract_prompt_template": "空压机设备能效对标",
    "chiller_efficiency_benchmark_extract_prompt_template": "冷水机组设备能效对标",
    "motor_efficiency_benchmark_extract_prompt_template": "电动机设备能效对标",
    "pump_efficiency_benchmark_extract_prompt_template": "水泵设备能效对标",
    "fan_efficiency_benchmark_extract_prompt_template": "风机设备能效对标",
    "boiler_efficiency_benchmark_extract_prompt_template": "锅炉或蒸汽发生器设备能效对标",
    "energy_metering_instrument_extract_prompt_template": "能源计量器具信息",
    "management_system_status_extract_prompt_template": "管理体系现状",
    "raw_material_consumption_extract_prompt_template": "主要原辅材料消耗",
    "green_material_usage_extract_prompt_template": "绿色物料使用",
    "energy_extract_prompt_template": "近三年能源消耗",
    "water_efficiency_metrics_extract_prompt_template": "水效指标信息",
    "waste_gas_treatment_extract_prompt_template": "废气治理设施",
    "waste_gas_emission_total_extract_prompt_template": "废气污染物排放总量情况",
    "wastewater_pollutants_extract_prompt_template": "废水污染物",
    "solid_waste_generation_disposal_extract_prompt_template": "近三年固体废物产生与处置",
    "noise_monitoring_extract_prompt_template": "噪音",
    "greenhouse_gas_emissions_extract_prompt_template": "温室气体",
    "land_intensification_extract_prompt_template": "用地集约化信息",
    "pollutant_emission_per_product_extract_prompt_template": "单位产品与产值主要污染物排放情况",
    "gas_emission_per_product_extract_prompt_template": "单位产品与产值主要废气排放情况",
    "wastewater_emission_per_product_extract_prompt_template": "单位产品与产值主要废水排放情况",
    "raw_material_consumption_per_product_extract_prompt_template": "单位产品与产值主要原材料消耗",
    "water_consumption_per_product_extract_prompt_template": "单位产品与产值水耗",
    "energy_consumption_per_product_extract_prompt_template": "单位产品与产值能耗",
    "carbon_emission_per_product_extract_prompt_template": "单位产品与产值主要碳排放情况",
    "green_production_evaluation_extract_prompt_template": "绿色生产水平综合评价",
    "qualification_certificate_info_extract_prompt_template": "资质证书信息",
    "tech_upgrade_scheme_summary_extract_prompt_template": "技改方案汇总与效益分析",
}


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


def _render_template_vars(template: str, **kwargs: str | None) -> str:
    values = {k: (v or "") for k, v in kwargs.items()}
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise HTTPException(status_code=400, detail=f"模板变量缺失: {missing}") from exc


def _preview_for_log(text: str, limit: int = 1200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _is_table_parent_chunk(hit: dict) -> bool:
    raw = str(hit.get("raw_content") or "")
    return (
        str(hit.get("label") or "") == "table"
        and raw.startswith("工作表：")
        and "粒度：sheet" in raw
    )


def _is_table_aggregate_chunk(hit: dict) -> bool:
    raw = str(hit.get("raw_content") or "")
    return (
        str(hit.get("label") or "") == "table"
        and raw.startswith("工作表：")
        and ("粒度：sheet" in raw or "粒度：window" in raw)
    )


def _is_table_row_chunk(hit: dict) -> bool:
    raw = str(hit.get("raw_content") or "")
    return (
        str(hit.get("label") or "") == "table"
        and '"工作表":"' in raw
        and '"行号":"' in raw
    )


def _expand_parent_hits(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits

    existing_ids = {str(hit.get("id") or "") for hit in hits}
    base_groups: dict[tuple[str, str], list[dict]] = {}
    for hit in hits:
        if str(hit.get("label") or "") != "table":
            continue
        file_id = str(hit.get("file_id") or "")
        source = str(hit.get("source") or "")
        if not file_id or not source:
            continue
        base_groups.setdefault((file_id, source), []).append(hit)

    parent_hits: list[dict] = []
    for (file_id, source), _group_hits in base_groups.items():
        try:
            related = Embedder.query_chunks_by_file(
                file_id=file_id,
                source=source,
                label="table",
                limit=64,
            )
        except Exception as exc:
            logger.warning("[Query] 回捞父块失败: file_id=%s err=%s", file_id[:12], exc)
            continue

        candidates = [
            item for item in related
            if _is_table_parent_chunk(item)
            and str(item.get("id") or "") not in existing_ids
        ]
        if not candidates:
            continue

        reranked = Reranker.rerank(query=query, hits=candidates, top_n=min(_PARENT_MAX_PER_FILE, len(candidates)))
        for parent in reranked:
            parent["parent_context"] = True
            parent_hits.append(parent)
            existing_ids.add(str(parent.get("id") or ""))

    if parent_hits:
        logger.info("[Query] Parent-child 补充父块 %d 条", len(parent_hits))
    return hits + parent_hits


def _prefer_table_aggregate_hits(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits

    replacement_by_group: dict[tuple[str, str], list[dict]] = {}
    for hit in hits:
        if not _is_table_row_chunk(hit):
            continue
        file_id = str(hit.get("file_id") or "")
        source = str(hit.get("source") or "")
        if not file_id or not source:
            continue
        replacement_by_group.setdefault((file_id, source), [])

    if not replacement_by_group:
        return hits

    replaced_groups = 0
    for file_id, source in replacement_by_group:
        try:
            related = Embedder.query_chunks_by_file(
                file_id=file_id,
                source=source,
                label="table",
                limit=64,
            )
        except Exception as exc:
            logger.warning("[Query] 查询表格聚合块失败: file_id=%s err=%s", file_id[:12], exc)
            continue

        aggregate_candidates = [item for item in related if _is_table_aggregate_chunk(item)]
        if not aggregate_candidates:
            continue

        reranked = Reranker.rerank(
            query=query,
            hits=aggregate_candidates,
            top_n=min(_PARENT_MAX_PER_FILE, len(aggregate_candidates)),
        )
        for item in reranked:
            item["table_context_preferred"] = True
        replacement_by_group[(file_id, source)] = reranked
        replaced_groups += 1

    if not replaced_groups:
        return hits

    merged_hits: list[dict] = []
    inserted_groups: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    for hit in hits:
        file_id = str(hit.get("file_id") or "")
        source = str(hit.get("source") or "")
        group_key = (file_id, source)

        if _is_table_row_chunk(hit) and group_key in replacement_by_group:
            if group_key in inserted_groups:
                continue
            inserted_groups.add(group_key)
            for replacement in replacement_by_group[group_key]:
                replacement_id = str(replacement.get("id") or "")
                if replacement_id and replacement_id in seen_ids:
                    continue
                if replacement_id:
                    seen_ids.add(replacement_id)
                merged_hits.append(replacement)
            continue

        hit_id = str(hit.get("id") or "")
        if hit_id and hit_id in seen_ids:
            continue
        if hit_id:
            seen_ids.add(hit_id)
        merged_hits.append(hit)

    logger.info("[Query] 表格聚合替换已生效: %d 个文件优先返回 sheet/window", replaced_groups)
    return merged_hits


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
    task_id: str | None = Field(None, description="限定任务范围")
    company_name: str | None = Field(None, description="企业名称（可用于 prompt 变量，不传则按 task_id 自动获取）")
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
    task_id: str | None,
    doc_type: str | None,
    file_id: str | None,
    prefer_source: str | None = None,
) -> tuple[list[dict], bool]:
    if doc_type and doc_type not in DocType.ALL:
        raise HTTPException(status_code=400, detail=f"无效的 doc_type: {doc_type}")

    filter_expr = _build_filter(task_id, doc_type, file_id)
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

    if prefer_source:
        keyword = prefer_source.strip().lower()
        if keyword:
            preferred_hits = [
                h for h in hits
                if keyword in ((h.get("source") or "").split("/")[-1].lower())
            ]
            if preferred_hits:
                logger.info(
                    "[Query] prefer_source 命中 %d/%d，启用文件内硬过滤: %r",
                    len(preferred_hits),
                    len(hits),
                    prefer_source,
                )
                hits = preferred_hits
            else:
                logger.info("[Query] prefer_source 未命中，回退全量候选: %r", prefer_source)

    logger.info("[Query] 向量粗筛: %d 条候选", len(hits))
    reranker_used = Reranker.is_available()
    rerank_top_n = len(hits) if prefer_source else top_k
    hits = Reranker.rerank(query=q, hits=hits, top_n=rerank_top_n)
    hits = _expand_parent_hits(q, hits)
    hits = _prefer_table_aggregate_hits(q, hits)
    if prefer_source:
        keyword = prefer_source.strip().lower()
        if keyword:
            # 双保险：在硬过滤后保留稳定排序逻辑
            hits.sort(key=lambda h: h.get("rerank_score", h.get("score", 0)), reverse=True)
        hits = hits[:top_k]
    logger.info("[Query] 精排后: %d 条", len(hits))
    _log_matched_sources(hits)
    return hits, reranker_used

@router.get("/query")
async def query(
    q: str = Query(..., min_length=1, description="问题"),
    top_k: int = Query(5, ge=1, le=20, description="最终返回条数（Reranker 精排后）"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="向量检索最低相似度"),
    task_id: str | None = Query(None, description="限定任务范围"),
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
        ?q=注册资本是多少&task_id=task_001
        ?q=2024年发票总金额&task_id=task_001&doc_type=invoice
    """
    logger.info("[Query] q=%r task=%s type=%s", q, task_id, doc_type)
    hits, reranker_used = _retrieve_hits(
        q=q,
        top_k=top_k,
        score_threshold=score_threshold,
        task_id=task_id,
        doc_type=doc_type,
        file_id=file_id,
        prefer_source=prefer_source,
    )

    if not hits:
        return JSONResponse({
            "query":      q,
            "task_id":    task_id,
            "answer":     "未找到相关内容，请确认文件已入库或放宽过滤条件。",
            "sources":    [],
            "chunks":     [],
            "reranker":   False,
            "llm_usage":  _llm_usage_payload(),
        })

    # ── STEP 3: 拼装 context ─────────────────────────────────────────
    context = _build_context(hits)
    logger.debug("[Query] context 长度: %d 字符", len(context))

    # ── STEP 4: 获取企业名称 ─────────────────────────────────────────
    company_name: str | None = None
    if task_id:
        company = Dedup.get_company(task_id)
        if company:
            company_name = company["company_name"]

    # ── STEP 5: 调 LLM ───────────────────────────────────────────────
    answer, llm_usage = await _call_llm(q, context, company_name=company_name)

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
                "task_id":       hit.get("task_id"),
                "score":         hit.get("score"),
                "rerank_score":  hit.get("rerank_score"),
            })

    logger.info("[Query] 答案 %d 字符，来源 %d 个，reranker=%s",
                len(answer), len(sources), reranker_used)

    return JSONResponse({
        "query":      q,
        "task_id":    task_id,
        "answer":     answer,
        "sources":    sources,
        "chunks":     [_public_hit(hit) for hit in hits],
        "reranker":   reranker_used,
        "llm_usage":  llm_usage,
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
        body.task_id,
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

    company_name = (body.company_name or "").strip() or None
    if not company_name and body.task_id:
        company = Dedup.get_company(body.task_id)
        if company:
            company_name = company.get("company_name")

    request_q = _render_template_vars(
        request_q,
        company_name=company_name,
        task_id=body.task_id,
    )
    request_system_prompt = _render_template_vars(
        request_system_prompt,
        company_name=company_name,
        task_id=body.task_id,
        text="",
    )
    request_user_prompt_template = _render_template_vars(
        request_user_prompt_template,
        company_name=company_name,
        task_id=body.task_id,
        text="{text}",
    )

    hits, reranker_used = _retrieve_hits(
        q=request_q,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        task_id=body.task_id,
        doc_type=body.doc_type,
        file_id=body.file_id,
        prefer_source=body.prefer_source,
    )

    if not hits:
        return JSONResponse({
            "query": request_q,
            "task_id": body.task_id,
            "structured_data": None,
            "raw_output": None,
            "sources": [],
            "chunks": [],
            "reranker": False,
            "message": "未找到相关内容，请确认文件已入库或放宽过滤条件。",
            "llm_usage": _llm_usage_payload(),
        })

    context = _build_context(hits)

    raw_output, structured_data, llm_usage = await _call_llm_extract(
        text=context,
        company_name=company_name,
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
                "task_id": hit.get("task_id"),
                "score": hit.get("score"),
                "rerank_score": hit.get("rerank_score"),
            })

    response: dict = {
        "query": request_q,
        "service_name": body.service_name,
        "service_cn_name": _SERVICE_CN_NAME_BY_NAME.get(body.service_name),
        "task_id": body.task_id,
        "company_name": company_name,
        "structured_data": structured_data,
        "sources": sources,
        "reranker": reranker_used,
        "llm_usage": llm_usage,
    }
    if body.include_chunks:
        response["chunks"] = [_public_hit(hit) for hit in hits]
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
        ordered_hits = sorted(
            group_hits,
            key=lambda h: (
                0 if h.get("parent_context") else 1,
                -float(h.get("rerank_score", h.get("score", 0)) or 0),
            ),
        )
        for hit in ordered_hits:
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
) -> tuple[str, dict]:
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
        started_at = time.perf_counter()
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
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        answer = resp.choices[0].message.content or ""
        prompt_tokens, completion_tokens = _extract_usage(resp)
        logger.debug("[LLM] 完成，tokens=%s", resp.usage)
        return answer.strip(), _llm_usage_payload(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
        )

    except Exception as exc:
        logger.error("[LLM] 调用失败: %s", exc, exc_info=True)
        return (
            f"（LLM 暂不可用，以下为原始检索内容）\n\n{context[:1500]}",
            _llm_usage_payload(),
        )


async def _call_llm_extract(
    *,
    text: str,
    company_name: str | None,
    system_prompt: str,
    user_prompt_template: str,
    max_tokens: int,
) -> tuple[str, dict | list | None, dict]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    client = AsyncOpenAI(
        base_url=settings.llm_api_base or "http://localhost:8000/v1",
        api_key=settings.llm_api_key or "EMPTY",
    )
    user_prompt = user_prompt_template.format(
        text=text,
        company_name=company_name or "",
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0
    has_usage = False
    started_at = time.perf_counter()

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
    prompt_tokens, completion_tokens = _extract_usage(resp)
    if prompt_tokens is not None or completion_tokens is not None:
        total_prompt_tokens += prompt_tokens or 0
        total_completion_tokens += completion_tokens or 0
        has_usage = True
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
            retry_prompt_tokens, retry_completion_tokens = _extract_usage(retry_resp)
            if retry_prompt_tokens is not None or retry_completion_tokens is not None:
                total_prompt_tokens += retry_prompt_tokens or 0
                total_completion_tokens += retry_completion_tokens or 0
                has_usage = True
            logger.debug(
                "[LLMExtract] retry finish_reason=%s 原始输出: %s",
                retry_finish_reason,
                retry_raw[:500],
            )
            retry_parsed = _safe_parse_json(retry_raw)
            if retry_parsed is not None:
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                return retry_raw, retry_parsed, _llm_usage_payload(
                    prompt_tokens=total_prompt_tokens if has_usage else None,
                    completion_tokens=total_completion_tokens if has_usage else None,
                    duration_ms=duration_ms,
                )
        except Exception as exc:
            logger.warning("[LLMExtract] 重试失败: %s", exc)

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return raw, parsed, _llm_usage_payload(
        prompt_tokens=total_prompt_tokens if has_usage else None,
        completion_tokens=total_completion_tokens if has_usage else None,
        duration_ms=duration_ms,
    )


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
        
