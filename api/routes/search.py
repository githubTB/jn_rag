"""
api/routes/search.py — 搜索接口。

接口
----
GET  /api/search   纯向量检索，返回相关 chunk 列表
GET  /api/query    向量检索 + Qwen 27B 生成答案（带来源标注）
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from config.settings import settings
from core.embedder import Embedder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


# ---------------------------------------------------------------------------
# GET /api/search — 纯向量检索
# ---------------------------------------------------------------------------

@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(5, ge=1, le=20, description="返回条数"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="最低相似度"),
    file_id: str | None = Query(None, description="限定文件范围（可选）"),
):
    """
    纯向量检索，返回最相关的 chunk 列表，不经过 LLM。

    适合：前端展示相关文档、调试检索效果。
    """
    logger.info("[Search] query=%r  top_k=%d  threshold=%.2f", q, top_k, score_threshold)

    filter_expr = f'file_id == "{file_id}"' if file_id else None

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
        "query":   q,
        "total":   len(hits),
        "results": hits,
    })


# ---------------------------------------------------------------------------
# GET /api/query — 向量检索 + LLM 生成答案
# ---------------------------------------------------------------------------

@router.get("/query")
async def query(
    q: str = Query(..., min_length=1, description="问题"),
    top_k: int = Query(5, ge=1, le=20, description="检索条数"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0, description="最低相似度"),
    file_id: str | None = Query(None, description="限定文件范围（可选）"),
):
    """
    向量检索 + Qwen 27B 生成带来源标注的答案。

    返回：
    - answer  : LLM 生成的答案
    - sources : 引用的 chunk 来源列表
    - chunks  : 检索到的原始 chunk（供前端展示）
    """
    logger.info("[Query] q=%r  top_k=%d", q, top_k)

    # ── STEP 1: 向量检索 ──────────────────────────────────────────────
    filter_expr = f'file_id == "{file_id}"' if file_id else None

    try:
        hits = Embedder.search(
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_expr=filter_expr,
        )
    except Exception as exc:
        logger.error("[Query] 检索失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}")

    if not hits:
        return JSONResponse({
            "query":   q,
            "answer":  "未找到相关内容，请换个问题或上传相关文档。",
            "sources": [],
            "chunks":  [],
        })

    # ── STEP 2: 拼装 context ──────────────────────────────────────────
    context_parts = []
    for i, hit in enumerate(hits, 1):
        source_name = hit.get("source", "未知来源").split("/")[-1]
        context_parts.append(
            f"[{i}] 来源：{source_name}（相似度 {hit['score']:.2f}）\n{hit['content']}"
        )
    context = "\n\n".join(context_parts)

    logger.debug("[Query] context 长度: %d 字符", len(context))

    # ── STEP 3: 调 LLM 生成答案 ───────────────────────────────────────
    answer = await _call_llm(q, context)

    # ── 来源去重（同一文件只列一次）──────────────────────────────────
    seen = set()
    sources = []
    for hit in hits:
        src = hit.get("source", "")
        if src not in seen:
            seen.add(src)
            sources.append({
                "file":  src.split("/")[-1],
                "path":  src,
                "score": hit["score"],
            })

    logger.info("[Query] 答案长度: %d 字符，来源: %d 个", len(answer), len(sources))

    return JSONResponse({
        "query":   q,
        "answer":  answer,
        "sources": sources,
        "chunks":  hits,
    })


# ---------------------------------------------------------------------------
# LLM 调用（OpenAI 兼容接口，Qwen 27B）
# ---------------------------------------------------------------------------

async def _call_llm(question: str, context: str) -> str:
    """
    调用 Qwen 27B 生成答案。
    使用 OpenAI 兼容接口，settings 里配好 LLM_API_BASE 和 LLM_API_KEY 即可。
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    client = AsyncOpenAI(
        base_url=settings.llm_api_base or "http://localhost:8002/v1",
        api_key=settings.llm_api_key or "EMPTY",
    )

    system_prompt = (
        "你是一个专业的文档问答助手。"
        "请根据下方提供的文档内容回答问题，答案要简洁准确。"
        "如果文档内容不足以回答问题，请明确说明。"
        "回答时用 [序号] 标注引用来源，例如：根据资料 [1] 显示..."
    )

    user_prompt = f"""文档内容：
{context}

问题：{question}"""

    logger.debug("[LLM] 调用 %s，model=%s", settings.llm_api_base, settings.llm_model)

    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = resp.choices[0].message.content or ""
        logger.debug("[LLM] 生成完成，tokens: %s", resp.usage)
        return answer.strip()

    except Exception as exc:
        logger.error("[LLM] 调用失败: %s", exc, exc_info=True)
        # LLM 失败时降级：直接返回检索到的内容
        fallback = f"（LLM 不可用，以下为原始检索内容）\n\n{context[:1000]}"
        return fallback