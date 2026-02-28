import json
from fastapi import APIRouter, HTTPException, Depends
import requests

from lakeflow.api.schemas.search import (
    EmbedRequest,
    EmbedResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    QARequest,
    QAResponse,
    QADebugInfo,
)

from lakeflow.common.text_normalizer import canonicalize_text
from lakeflow.services.ollama_embed_service import embed_batch, OLLAMA_EMBED_URL, EMBED_MODEL
from lakeflow.core.auth import verify_token
from lakeflow.catalog.app_db import insert_message
from lakeflow.vectorstore.constants import COLLECTION_NAME as DEFAULT_COLLECTION_NAME
from lakeflow.core.config import get_qdrant_url, LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY
from lakeflow.services.llm_chat_service import chat_completion, USE_OLLAMA_NATIVE_CHAT

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Vector hóa chuỗi",
    description="Trả về vector embedding của một chuỗi (dùng cùng model với semantic search).",
)
def embed_text(req: EmbedRequest) -> dict:
    vector = embed_batch([req.text])[0]
    return {
        "text": req.text,
        "vector": vector,
        "embedding": vector,
        "dim": len(vector),
    }


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
)
def semantic_search(req: SemanticSearchRequest):
    """
    Semantic search dùng Qdrant REST API (requests)
    """

    # --------------------------------------------------
    # 1. Embed query
    # --------------------------------------------------
    
    expanded_query = canonicalize_text(req.query)
    query_vector = embed_batch([expanded_query])[0]

    # --------------------------------------------------
    # 2. Call Qdrant REST API
    # --------------------------------------------------
    base = get_qdrant_url(req.qdrant_url)
    coll = (req.collection_name or DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME
    url = f"{base}/collections/{coll}/points/search"

    payload = {
        "vector": query_vector,
        "limit": req.top_k,
        "with_payload": True,
        "with_vector": False,
    }
    if req.score_threshold is not None:
        payload["score_threshold"] = req.score_threshold

    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Qdrant search failed: {exc}")

    data = resp.json()

    # --------------------------------------------------
    # 3. Parse response
    # --------------------------------------------------
    points = data.get("result", [])

    results = []
    for p in points:
        pl = p.get("payload", {}) or {}
        results.append({
            "id": p.get("id"),
            "score": float(p.get("score", 0.0)),
            "file_hash": pl.get("file_hash"),
            "chunk_id": pl.get("chunk_id"),
            "section_id": pl.get("section_id"),
            "text": pl.get("text"),
            "token_estimate": pl.get("token_estimate"),
            "source": pl.get("source"),
        })

    return {
        "query": req.query,
        "results": results,
    }


def _curl_multiline(url: str, payload_obj: dict, *, auth: bool = False) -> str:
    """Tạo lệnh curl đa dòng: curl URL \\ -H ... \\ -d '{...}'"""
    payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
    payload_escaped = payload.replace("'", "'\"'\"'")
    lines = [f"curl '{url}' \\", '  -H "Content-Type: application/json" \\']
    if auth:
        lines.append('  -H "Authorization: Bearer $OPENAI_API_KEY" \\')
    lines.append(f"  -d '{payload_escaped}'")
    return "\n".join(lines)


def _curl_embed(question: str) -> str:
    payload_obj = {"model": EMBED_MODEL, "input": [question]}
    return _curl_multiline(OLLAMA_EMBED_URL, payload_obj)


def _curl_search(qdrant_base: str, coll: str, vector: list, top_k: int, score_threshold: float | None) -> str:
    body: dict = {"vector": vector, "limit": top_k, "with_payload": True, "with_vector": False}
    if score_threshold is not None:
        body["score_threshold"] = score_threshold
    url = f"{qdrant_base.rstrip('/')}/collections/{coll}/points/search"
    return _curl_multiline(url, body)


def _curl_complete(messages: list, temperature: float, max_tokens: int) -> str:
    """Curl complete: format chuẩn, khớp endpoint backend đang dùng."""
    base = LLM_BASE_URL.rstrip("/")
    if USE_OLLAMA_NATIVE_CHAT:
        url = f"{base}/api/chat"
        payload_obj = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 1000},
        }
    else:
        url = f"{base}/v1/chat/completions"
        payload_obj = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    return _curl_multiline(url, payload_obj, auth=bool(OPENAI_API_KEY))


@router.post(
    "/qa",
    response_model=QAResponse,
)
def qa(req: QARequest, auth_payload: dict = Depends(verify_token)):
    """
    Q&A với RAG: Tìm context từ semantic search, sau đó dùng LLM để trả lời.
    Tin nhắn (câu hỏi) được ghi theo username để thống kê trong Admin.
    Trả về debug_info gồm curl commands và tiến độ các bước.
    """
    steps_done: list[str] = []
    curl_embed = _curl_embed(req.question)

    # --------------------------------------------------
    # 1. Embed query (vector hóa câu hỏi)
    # --------------------------------------------------
    expanded_query = canonicalize_text(req.question)
    try:
        query_vector = embed_batch([expanded_query])[0]
    except (RuntimeError, requests.RequestException) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bước 1 (Embed) thất bại. Curl để test:\n{curl_embed}\n\nLỗi: {exc}"
        )

    steps_done.append("1. Embed câu hỏi (Ollama)")
    base = get_qdrant_url(req.qdrant_url)
    coll = (req.collection_name or DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME
    url = f"{base}/collections/{coll}/points/search"
    curl_search = _curl_search(base, coll, query_vector, req.top_k, req.score_threshold)

    search_body = {
        "vector": query_vector,
        "limit": req.top_k,
        "with_payload": True,
        "with_vector": False,
    }
    if req.score_threshold is not None:
        search_body["score_threshold"] = req.score_threshold

    try:
        resp = requests.post(
            url,
            json=search_body,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bước 2 (Qdrant Search) thất bại. Curl để test:\n{curl_search}\n\nLỗi: {exc}"
        )

    steps_done.append("2. Tìm context (Qdrant)")
    data = resp.json()
    points = data.get("result", [])

    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"Bước 2 hoàn thành nhưng không tìm thấy context. Curl đã chạy:\n{curl_search}\n\nKhông tìm thấy document nào phù hợp."
        )

    # Parse context results
    contexts = []
    context_texts = []

    for p in points:
        pl = p.get("payload", {}) or {}
        context_text = pl.get("text", "")
        contexts.append({
            "id": p.get("id"),
            "score": float(p.get("score", 0.0)),
            "file_hash": pl.get("file_hash"),
            "chunk_id": pl.get("chunk_id"),
            "section_id": pl.get("section_id"),
            "text": context_text,
            "token_estimate": pl.get("token_estimate"),
            "source": pl.get("source"),
        })
        
        if context_text:
            context_texts.append(context_text)
    
    # --------------------------------------------------
    # 2. Build prompt với context
    # --------------------------------------------------
    context_block = "\n\n".join([
        f"[Context {i+1}]:\n{text}"
        for i, text in enumerate(context_texts)
    ])
    
    system_prompt = """Bạn đang tham gia một demo RAG (Retrieval-Augmented Generation). Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên các đoạn tài liệu (context) được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
- Chỉ được trả lời dựa trên nội dung trong context. Không dùng kiến thức bên ngoài, không suy đoán thêm.
- Nếu câu trả lời có trong context: trích dẫn hoặc tóm tắt từ context một cách chính xác, trả lời bằng tiếng Việt.
- Nếu context không chứa thông tin để trả lời câu hỏi: hãy nói rõ "Theo các tài liệu được cung cấp, không có thông tin để trả lời câu hỏi này." và không bịa đáp án.
- Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt."""

    user_prompt = f"""Các đoạn tài liệu (context) dùng để trả lời — CHỈ dựa vào đây:

{context_block}

---
Câu hỏi: {req.question}

Trả lời (chỉ dựa trên context trên):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    curl_complete = _curl_complete(messages, req.temperature, 1000)

    # --------------------------------------------------
    # 3. Gọi LLM (OpenAI-compatible hoặc native Ollama /api/chat)
    # --------------------------------------------------
    try:
        answer, model_used = chat_completion(
            messages=messages,
            temperature=req.temperature,
            max_tokens=1000,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bước 3 (LLM Complete) thất bại. Curl để test:\n{curl_complete}\n\nLỗi: {exc}",
        )
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bước 3 (LLM Complete) — phản hồi không hợp lệ. Curl đã chạy:\n{curl_complete}\n\nLỗi: {exc}",
        )

    steps_done.append("3. Gọi LLM (Complete)")

    # Ghi tin nhắn theo user (để thống kê / xóa trong Admin)
    try:
        insert_message(username=auth_payload["sub"], question=req.question)
    except Exception:
        pass  # Không làm fail request Q&A nếu ghi DB lỗi

    return {
        "question": req.question,
        "answer": answer,
        "contexts": contexts,
        "model_used": model_used,
        "debug_info": QADebugInfo(
            steps_completed=steps_done,
            curl_embed=curl_embed,
            curl_search=curl_search,
            curl_complete=curl_complete,
        ),
    }
