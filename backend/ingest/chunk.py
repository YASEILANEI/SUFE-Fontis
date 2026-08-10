"""按章节/条目切分文档文本，产出带来源标注的分块，并做敏感信息过滤。

产出：
  backend/data/chunks.json   [{id, doc_id, title, category, rel_path, heading, text, idx}]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .scan_sensitive import filter_chunks

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
PARSED_DIR = DATA_DIR / "parsed"

MAX_LEN = 700  # 单块目标字数

# 章节/条目锚点：第X章/第X条/一、/1./（一）等
ANCHOR_RE = re.compile(
    r"^(第[一二三四五六七八九十百\d]+[章节部分条条款]|"
    r"[一二三四五六七八九十百]+、|\d{1,2}[\.、]|"
    r"（[一二三四五六七八九十]+）)\s*(\S.*)?$"
)


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ANCHOR_RE.match(line):
            if lines:
                sections.append((heading, lines))
            heading = line
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    return sections


def split_long(text_lines: list[str]) -> list[str]:
    """超长章节按行累积切成 ~MAX_LEN 子块。"""
    out: list[str] = []
    buf: list[str] = []
    cur = 0
    for line in text_lines:
        cur += len(line)
        buf.append(line)
        if cur >= MAX_LEN:
            out.append("\n".join(buf))
            buf, cur = [], 0
    if buf:
        out.append("\n".join(buf))
    return out


def chunk_document(doc: dict) -> list[dict]:
    safe_id = doc["id"].replace("/", "__")
    text = (PARSED_DIR / f"{safe_id}.txt").read_text(encoding="utf-8")
    chunks: list[dict] = []
    idx = 0
    for heading, lines in split_sections(text):
        for block in split_long(lines):
            if not block.strip():
                continue
            chunks.append(
                {
                    "id": f'{doc["id"]}#{idx}',
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "rel_path": doc["rel_path"],
                    "heading": heading,
                    "text": block.strip(),
                    "idx": idx,
                }
            )
            idx += 1
    return chunks


def main() -> None:
    docs = json.loads((DATA_DIR / "documents.json").read_text(encoding="utf-8"))
    all_chunks: list[dict] = []
    for doc in docs:
        if doc.get("note") == "scan":
            print(f"跳过扫描件（无文字层，需 OCR）：{doc['title']}")
            continue
        all_chunks.extend(chunk_document(doc))

    kept, dropped = filter_chunks(all_chunks)
    (DATA_DIR / "chunks.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"分块完成：{len(kept)} 块（{len(dropped)} 块因敏感信息被过滤）")
    if dropped:
        for d in dropped[:20]:
            print(f"  过滤 [{d['title']}] {d.get('sensitive')}：{d['text'][:40]}...")
        if len(dropped) > 20:
            print(f"  ... 共 {len(dropped)} 块")


if __name__ == "__main__":
    main()
