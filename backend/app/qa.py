"""LLM 问答：检索片段 + dsv4f（OpenAI 兼容接口）生成带引用回答。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]  # backend/
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "dsv4f")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not API_KEY or not BASE_URL:
            raise RuntimeError("未配置 OPENAI_API_KEY / OPENAI_BASE_URL（见 backend/.env.example）")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


SYSTEM_PROMPT = """你是"上财校园助手"，服务于上海财经大学学生，帮助他们查询学校的规章制度与办事流程。

回答规则：
1. 你的全部知识来自下方提供的官方资料片段，除此之外不得使用任何其他信息，禁止编造。
2. 若资料片段不足以回答，直接说明"资料中未找到相关信息"，可提示用户该问题可能涉及的文件名称。
3. 回答简洁清楚，用中文；涉及数字、条件、时限时尽量引用原文表述，不自行解读。
4. 在回答中依据片段编号标注来源，形如 [来源1]、[来源2]。"""


def build_user_prompt(chunks: list[dict], question: str) -> str:
    refs = []
    for i, c in enumerate(chunks, 1):
        heading = f"，章节：{c['heading']}" if c.get("heading") else ""
        refs.append(f"[{i}]（文件：{c['title']}{heading}）\n{c['text']}")
    return (
        "官方资料片段：\n\n"
        + "\n\n".join(refs)
        + f"\n\n学生问题：{question}\n\n请依据上述片段回答，并标注来源。"
    )


def sources_of(chunks: list[dict], max_text: int = 300) -> list[dict]:
    out = []
    for c in chunks:
        text = c["text"].strip()
        out.append(
            {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "category": c["category"],
                "rel_path": c["rel_path"],
                "heading": c.get("heading", ""),
                "text": text[:max_text],
            }
        )
    return out


def stream_answer(chunks: list[dict], question: str, history: list[dict] | None = None):
    """返回 delta 文本生成器。history 为最近对话轮次，用于多轮上下文。"""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # 只取最近若干轮，避免上下文过长
        messages += [{"role": m["role"], "content": m["content"]}
                     for m in history[-8:] if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": build_user_prompt(chunks, question)})

    client = get_client()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
