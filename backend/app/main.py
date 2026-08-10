"""FastAPI 入口：/api/chat（SSE 流式）、/api/meta、/api/doc，及静态前端托管。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .qa import sources_of, stream_answer
from .search import Retriever

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
PARSED_DIR = DATA_DIR / "parsed"
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="上财校园助手")

_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 从比赛通知整理的高频问题（首页引导）
SUGGESTED_QUESTIONS = [
    "推免（保研）需要满足哪些条件？",
    "本科生奖学金评选有哪些要求？",
    "差旅住宿费报销的限额是多少？",
    "毕业论文重复率检测的要求是什么？",
    "研究生学位论文双盲评审怎么进行？",
    "发展党员的基本流程是什么？",
]


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []  # 最近对话轮次，[{role, content}], 可选
    category: str = ""  # 限定知识库分类检索，空为全库


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


@app.get("/api/doc/{doc_id}")
def doc(doc_id: str) -> dict:
    path = PARSED_DIR / f"{doc_id}.txt"
    if not path.exists():
        return {"error": "未找到该文档"}
    return {"id": doc_id, "text": path.read_text(encoding="utf-8")}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        return StreamingResponse(iter([sse("error", {"message": "问题不能为空"})]),
                                 media_type="text/event-stream")

    chunks = retriever().retrieve(question, category=req.category or None)
    if not chunks:
        return StreamingResponse(iter([sse("error", {"message": "未检索到相关资料，请换个问法"})]),
                                 media_type="text/event-stream")

    def gen():
        sources = [
            {**s, "safe_id": s["doc_id"].replace("/", "__")}
            for s in sources_of(chunks)
        ]
        yield sse("sources", {"sources": sources})
        try:
            for delta in stream_answer(chunks, question, req.history):
                yield sse("delta", {"text": delta})
            yield sse("done", None)
        except Exception as exc:
            yield sse("error", {"message": f"生成回答失败：{exc}"})

    return StreamingResponse(gen(), media_type="text/event-stream")
