"""FastAPI 入口：/api/chat（SSE 流式）、/api/meta、/api/doc，及静态前端托管。"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .citations import compute_citations
from .qa import build_retrieval_query, rerank_chunks, rewrite_query, sources_of, stream_answer
from .search import Retriever

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
PARSED_DIR = DATA_DIR / "parsed"
DOCS_DIR = ROOT / "规章制度"
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="金匮问渠 · SUFE Fontis")

_retriever: Retriever | None = None
_documents_by_safe: dict[str, dict] | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 首页引导问题（按分类组织，供前端分组展示）
SUGGESTED_QUESTIONS = [
    {"category": "学业", "questions": [
        "推免（保研）需要满足哪些条件？",
        "挂科了还能不能顺利毕业？",
        "毕业论文重复率超过多少需要修改？",
        "转专业需要什么条件？",
    ]},
    {"category": "研究生", "questions": [
        "研究生学位论文双盲评审怎么进行？",
        "研究生怎么申请出国交流项目？",
    ]},
    {"category": "财务", "questions": [
        "差旅住宿费报销的限额是多少？",
        "学费应该怎么缴纳？",
    ]},
    {"category": "出国出境", "questions": [
        "本科生可以申请哪些出国（境）项目？",
    ]},
    {"category": "学工", "questions": [
        "本科生奖学金评选有哪些要求？",
    ]},
    {"category": "党建", "questions": [
        "发展党员的基本流程是什么？",
    ]},
]


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []  # 最近对话轮次，[{role, content}], 可选
    category: str = ""  # 限定知识库分类检索，空为全库


MAX_QUESTION = 500            # 问题最长字符数（防滥用/防超长上下文）
MAX_CATEGORY = 50             # 分类参数最长字符数
MAX_HISTORY = 20              # 历史最多携带条数
MAX_HISTORY_CONTENT = 2000    # 单条历史内容最长字符数


def validate_chat_input(question: str, category: str) -> str | None:
    """校验输入长度，返回错误消息；None 表示通过。question 需已 strip。"""
    if not question:
        return "问题不能为空"
    if len(question) > MAX_QUESTION:
        return f"问题过长（最多 {MAX_QUESTION} 字）"
    if len(category) > MAX_CATEGORY:
        return "分类参数过长"
    return None


def normalize_history(history: list[dict] | None) -> list[dict]:
    """裁剪历史输入：只保留 user/assistant 消息，单条内容截断，最多保留最近 MAX_HISTORY 条。"""
    if not history:
        return []
    cleaned: list[dict] = []
    for m in history[-MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": str(content)[:MAX_HISTORY_CONTENT]})
    return cleaned


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/api/meta")
def meta() -> dict:
    r = retriever()
    cats = Counter(c["category"] for c in r.chunks)
    return {
        "doc_count": len({c["doc_id"] for c in r.chunks}),
        "chunk_count": len(r.chunks),
        "vector": r.status.get("vector", False),
        "categories": [{"name": k, "count": v} for k, v in sorted(cats.items())],
        "questions": SUGGESTED_QUESTIONS,
    }


def _valid_safe_id(doc_id: str) -> bool:
    return bool(doc_id) and doc_id not in {".", ".."} and "/" not in doc_id and "\\" not in doc_id


def _documents() -> dict[str, dict]:
    global _documents_by_safe
    if _documents_by_safe is None:
        source = DATA_DIR / "documents.json"
        records = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
        _documents_by_safe = {
            record["id"].replace("/", "__"): record
            for record in records
            if record.get("id") and record.get("rel_path")
        }
    return _documents_by_safe


def _source_path(doc_id: str) -> Path | None:
    if not _valid_safe_id(doc_id):
        return None
    path = (PARSED_DIR / f"{doc_id}.txt").resolve()
    try:
        path.relative_to(PARSED_DIR.resolve())
    except ValueError:
        return None
    return path


@app.get("/api/doc/{doc_id}/download")
def download_doc(doc_id: str) -> FileResponse:
    if not _valid_safe_id(doc_id):
        raise HTTPException(status_code=404, detail="未找到该文档")
    record = _documents().get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="未找到该文档")
    source = (ROOT / record["rel_path"]).resolve()
    try:
        source.relative_to(DOCS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="原始文件不存在")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="原始文件不存在")
    return FileResponse(source, filename=source.name)


@app.get("/api/doc/{doc_id}")
def doc(doc_id: str, cite: str = "") -> dict:
    path = _source_path(doc_id)
    if path is None or not path.exists():
        return {"error": "未找到该文档"}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": "该文档编码无法识别，暂时无法显示原文"}
    except OSError:
        return {"error": "读取文档失败"}
    if not text.strip():
        return {"error": "该文档暂无文本内容（可能是扫描件），无法显示原文"}
    result = {"id": doc_id, "text": text}
    if cite:
        cite_ids = [item for item in cite.split(",") if item]
        result["citations"] = compute_citations(text, retriever().by_id, cite_ids)
    return result


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    error = validate_chat_input(question, req.category)
    if error:
        return StreamingResponse(iter([sse("error", {"message": error})]),
                                 media_type="text/event-stream")
    history = normalize_history(req.history)

    try:
        base_query = build_retrieval_query(history, question)
        retrieval_query = rewrite_query(base_query)
        # 多路召回：原始词与改写词各检索 top 12 合并去重，避免改写词单路偏科导致召回不全
        merged: dict[str, dict] = {}
        for c in retriever().retrieve(base_query, k=12, category=req.category or None):
            merged[c["id"]] = c
        for c in retriever().retrieve(retrieval_query, k=12, category=req.category or None):
            merged.setdefault(c["id"], c)
        candidates = list(merged.values())
        chunks = rerank_chunks(base_query, candidates, top_k=6)
    except Exception:
        logging.getLogger("uvicorn.error").exception("检索失败")
        return StreamingResponse(iter([sse("error", {"message": "检索失败，请稍后重试"})]),
                                 media_type="text/event-stream")

    def gen():
        sources = [
            {**s, "safe_id": s["doc_id"].replace("/", "__")}
            for s in sources_of(chunks)
        ]
        yield sse("sources", {"sources": sources})
        try:
            for delta in stream_answer(chunks, question, history):
                yield sse("delta", {"text": delta})
            yield sse("done", None)
        except Exception:
            logging.getLogger("uvicorn.error").exception("生成回答失败")
            yield sse("error", {"message": "生成回答失败，请稍后重试"})

    return StreamingResponse(gen(), media_type="text/event-stream")
