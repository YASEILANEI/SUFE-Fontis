"""端到端评估：跑一批问题，走真实完整链路（AI 改写→BM25→AI 重排→流式回答），输出 UTF-8 报告。

用法（在 backend/ 下，用 .venv python 执行，需 .env 配好 API key）：
  python scripts/eval_queries.py                          # 跑默认问题集（挂科问 2 次）
  python scripts/eval_queries.py --query "毕业论文重复率"   # 只跑一个问题
  python scripts/eval_queries.py --history "问1|问2"       # 带多轮历史（追问场景）
  python scripts/eval_queries.py --out data/eval_report.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from app.qa import build_retrieval_query, rerank_chunks, rewrite_query, stream_answer
from app.search import Retriever

DEFAULT_QUERIES = [
    ("常识判断", "挂科了还能不能顺利毕业"),
    ("常识判断", "挂科会不会影响保研和评奖"),
    ("常识判断", "转专业需要什么条件"),
    ("精确规定", "毕业论文重复率超过多少需要修改"),
    ("精确规定", "差旅住宿费报销的限额是多少"),
    ("精确规定", "推免（保研）需要满足哪些条件"),
    ("库外", "校医院怎么挂号"),
    ("库外", "校园卡丢了应该怎么办"),
]

REPEAT = {"挂科了还能不能顺利毕业": 2}

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def run_one(question: str, history: list[dict] | None = None) -> dict:
    t0 = time.time()
    base = build_retrieval_query(history, question)
    rewrote = rewrite_query(base)
    cand = get_retriever().retrieve(rewrote, k=18)
    chunks = rerank_chunks(base, cand, top_k=6)
    answer = "".join(stream_answer(chunks, question, history))
    return {
        "question": question,
        "rewrote": rewrote,
        "sources": [
            {
                "title": c["title"],
                "doc_id": c["doc_id"],
                "heading": c.get("heading", ""),
                "score": c.get("score", 0),
            }
            for c in chunks
        ],
        "answer": answer,
        "elapsed": round(time.time() - t0, 1),
    }


def parse_history(s: str | None) -> list[dict] | None:
    if not s:
        return None
    return [{"role": "user", "content": q.strip()} for q in s.split("|") if q.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="只跑这一个问题")
    ap.add_argument("--history", help="多轮历史，user 消息用 | 分隔，如 \"问1|问2\"")
    ap.add_argument("--out", default=str(BACKEND / "data" / "eval_report.txt"))
    args = ap.parse_args()

    if args.query:
        runs = [("单问", args.query)]
    else:
        runs = []
        for cat, q in DEFAULT_QUERIES:
            runs.append((cat, q))
            for _ in range(REPEAT.get(q, 1) - 1):
                runs.append((cat, q))

    lines: list[str] = []
    for cat, q in runs:
        print(f"▶ [{cat}] {q} ...", flush=True)
        try:
            r = run_one(q, parse_history(args.history))
        except Exception as exc:
            lines.append(f"===== [{cat}] {q} =====\n[ERROR] {exc!r}\n")
            print(f"  ✗ ERROR {exc!r}", flush=True)
            continue
        lines.append(f"===== [{cat}] {q} =====")
        lines.append(f"耗时: {r['elapsed']}s")
        lines.append(f"改写检索词: {r['rewrote']}")
        lines.append("来源:")
        for i, s in enumerate(r["sources"], 1):
            lines.append(f"  [{i}] {s['doc_id']} | {s['title']} | score={s['score']} | {s['heading']}")
        lines.append("回答:")
        lines.append(r["answer"])
        lines.append("")
        print(f"  ✓ {len(r['answer'])} 字", flush=True)

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
