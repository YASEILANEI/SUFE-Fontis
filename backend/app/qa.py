"""LLM 问答：检索片段 + dsv4f（OpenAI 兼容接口）生成带引用回答。"""
from __future__ import annotations

import os
import re
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


SYSTEM_PROMPT = """你是"金匮问渠"，服务于上海财经大学学生的 AI 助教，帮助他们查询学校的规章制度与办事流程。

回答规则：
1. 你的全部知识来自下方提供的官方资料片段，除此之外不得使用任何其他信息，禁止编造。
2. 若资料片段不足以回答，直接说明"资料中未找到相关信息"，可提示用户该问题可能涉及的文件名称或部门。
3. 回答结构清晰、内容完整：当问题涉及多个方面时，用加粗小标题或分点分段组织，尽量覆盖片段中与问题相关的全部要点，不遗漏关键条件、时限、流程步骤。
4. 涉及数字、条件、时限、流程步骤等精确信息时，尽量引用原文表述，不自行解读、不凭空补充。
5. 在回答中依据片段编号标注来源，形如 [来源1]、[来源2]。
6. 对话可能是多轮的：用户最新的消息可能是追问、补充或澄清，请结合整个对话理解最新消息的意图，在之前的讨论背景下回答。"""


def build_retrieval_query(history: list[dict] | None, question: str) -> str:
    """把最近几轮对话拼进检索查询词，让追问/补充能定位回原话题。"""
    if not history:
        return question
    recent = []
    for m in history[-6:]:  # 最近 ~3 轮
        if m.get("role") != "user":
            continue
        c = (m.get("content") or "").strip()
        if c:
            recent.append(c[:200])  # 每条截断，避免噪音
    return " ".join(recent) + " " + question if recent else question


REWRITE_SYSTEM = """你是检索词改写器，负责把学生的口语提问改写成"上海财经大学官方规章制度文件"中会使用的检索关键词。

硬性要求：
1. 必须改写，禁止原样照抄用户的问题句子；即使用户句子已是书面语，也要拆解成若干检索关键词。
2. 输出 2~4 个关键词短语，用空格分隔，每个词都要尽量使用官方文件用语。
3. 口语必须映射为官方术语，例如：挂科→课程考核不合格；补考/重修→课程考核不及格；保研→推荐免试；贫困生→家庭经济困难学生；换导师→变更指导教师；奖学金→奖学金评定；毕业论文→毕业论文（设计）；报销→差旅费报销；转专业→转专业；校园卡→校园卡补办。
4. 保留关键限定词（如 推免、出国交流、期限、金额、比例、免试 等），但口语化的修饰词（"能不能""怎么""需要满足哪些""什么条件"）一律不要。
5. 多轮追问时结合上文锁定主题，只输出针对当前问题的检索词。
6. 只输出检索关键词本身，不要任何解释、引号、标点或编号。"""

_rewrite_cache: dict[str, str] = {}

_ORAL_WORDS = (
    "能不能", "怎么", "什么", "需要满足", "哪些", "需要", "可以",
    "是否", "咋", "吗", "呢", "要不要",
)


def _is_unchanged(rewritten: str, raw_query: str) -> bool:
    """模型原样返回或仍像口语问题时判为未改写，触发重试。"""
    if not rewritten or rewritten == raw_query:
        return True
    return any(w in rewritten for w in _ORAL_WORDS)


# 口语→官方术语映射：LLM 改写失败时的确定性兜底，保证 BM25 能命中官方条款
_SYNONYM_MAP = {
    "挂科": "课程考核不合格 补考 重修",
    "补考": "课程考核不及格 补考",
    "保研": "推荐免试",
    "评奖": "评奖评优",
    "评优": "评奖评优",
    "贫困生": "家庭经济困难学生",
    "补助": "资助",
    "换导师": "变更指导教师",
    "查重": "重复率检测",
    "校园卡": "校园卡补办 挂失",
    "考研": "硕士研究生招生 初试 复试",
    "宿舍": "学生公寓 住宿管理",
    "退课": "退选 课程修读",
    "选课": "选课 课程修读",
    "绩点": "绩点 成绩",
    "医保": "医疗保障 医疗费用",
    "一卡通": "校园卡",
    "饭卡": "校园卡",
    "毕业实习": "毕业实习",
    "转专业": "转专业 自主选择专业",
    "休学": "休学",
    "学费": "学费缴纳",
    "出国交换": "出国 出境 交流学习",
    "交换生": "出国 出境 交流学习",
    "双学位": "辅修 双学位",
    "辅修": "辅修",
    "助学金": "助学金 资助",
    "奖学金": "奖学金评定",
    "入党": "发展党员",
    "优秀党员": "党员 表彰",
    "报销": "差旅费报销 财务报销",
    "请假": "请假",
}


def expand_synonyms(query: str) -> str:
    """把 query 中出现的口语词扩展为官方术语，返回扩展后的检索词。"""
    extra = [official for oral, official in _SYNONYM_MAP.items() if oral in query]
    return query + " " + " ".join(extra) if extra else query


def _too_short(rewritten: str) -> bool:
    """改写结果过短（如只留 1-2 个词）视为不可用，如'校园卡'。"""
    return len(rewritten.replace(" ", "").replace("　", "")) < 6


def rewrite_query(raw_query: str) -> str:
    """LLM 改写口语提问为官方检索词；未改写/过短则重试，仍失败用词典扩展兜底。"""
    cached = _rewrite_cache.get(raw_query)
    if cached is not None:
        return cached
    for _ in range(2):
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": f"学生的问题：{raw_query}"},
                ],
                temperature=0.1,
                max_tokens=150,
                timeout=10,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
        except Exception:
            return expand_synonyms(raw_query)
        if not rewritten:
            return expand_synonyms(raw_query)
        if _is_unchanged(rewritten, raw_query) or _too_short(rewritten):
            continue  # 原样返回或过短，重试一次
        _rewrite_cache[raw_query] = rewritten
        return rewritten
    return expand_synonyms(raw_query)


RERANK_SYSTEM = """你是资料相关性判断器。学生提出问题，下面列出若干官方资料片段，请判断每个片段与该问题的相关性。

规则：
1. 只输出相关片段的编号，按相关度从高到低排列，用空格分隔。
2. 若没有任何相关片段，输出空内容，不要输出任何文字。
3. 不要输出解释、标点或其它内容。"""

_RERANK_TEXT_MAX = 220


def rerank_chunks(question: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
    """LLM 按与问题的相关性重排候选片段；任何失败回退原候选的前 top_k。"""
    if len(candidates) <= top_k:
        return candidates
    numbered = [
        f"[{i}] {c['text'].strip()[:_RERANK_TEXT_MAX]}" for i, c in enumerate(candidates, 1)
    ]
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": f"学生问题：{question}\n" + "\n".join(numbered)},
            ],
            temperature=0,
            max_tokens=40,
            timeout=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return candidates[:top_k]
    ids = [int(m) for m in re.findall(r"\d+", raw)]
    ordered = [candidates[i - 1] for i in ids if 1 <= i <= len(candidates)]
    if not ordered:
        return candidates[:top_k]
    return ordered[:top_k]


def build_user_prompt(chunks: list[dict], question: str) -> str:
    if not chunks:
        return (
            "本次未检索到相关官方资料片段。\n\n"
            f"学生最新问题：{question}\n\n"
            "若是普遍性常识问题，直接给出明确回答；若涉及上海财经大学的具体规定、数字、条款，"
            "请说明检索资料中未找到，并声明以学校正式文件或相关部门通知为准。"
        )
    refs = []
    for i, c in enumerate(chunks, 1):
        heading = f"，章节：{c['heading']}" if c.get("heading") else ""
        refs.append(f"[{i}]（文件：{c['title']}{heading}）\n{c['text']}")
    return (
        "官方资料片段：\n\n"
        + "\n\n".join(refs)
        + f"\n\n学生最新问题：{question}（若是追问请结合上文理解）\n\n"
        "回答须结构清晰、内容完整，覆盖片段中与问题相关的全部要点；"
        "涉及本校具体规定、流程、数字、条件、时限的，必须以片段为准并标注 [来源N]；"
        "片段不足时明确说明，不编造。"
    )


def sources_of(chunks: list[dict], max_text: int = 300) -> list[dict]:
    out = []
    for c in chunks:
        text = c["text"].strip()
        out.append(
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["id"],
                "title": c["title"],
                "category": c["category"],
                "rel_path": c["rel_path"],
                "heading": c.get("heading", ""),
                "text": text[:max_text],
                "file_ext": c["rel_path"].rsplit(".", 1)[-1].lower(),
                "score": c.get("score", 0),
            }
        )
    return out


def stream_answer(chunks: list[dict], question: str, history: list[dict] | None = None):
    """返回 delta 文本生成器。history 为最近对话轮次，用于多轮上下文。"""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # 只取最近若干轮，避免上下文过长
        messages += [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in history[-8:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
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
