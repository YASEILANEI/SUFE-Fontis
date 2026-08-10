"""构建检索索引：向量库（可选）+ BM25 关键词索引。

向量库依赖 sentence-transformers + chromadb（requirements-vector.txt）；
未安装时自动降级，仅构建 BM25，运行时检索仍可用。

产出：
  backend/data/chroma/            ChromaDB 持久化（向量可用时）
  backend/data/bm25_tokens.json   BM25 分词结果（运行时重建索引）
  backend/data/index_ready.json   检索索引状态
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")


def build_bm25(chunks: list[dict]) -> None:
    payload = []
    for c in chunks:
        payload.append({"id": c["id"], "tokens": jieba.lcut(c["text"])})
    (DATA_DIR / "bm25_tokens.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(f"BM25 索引：{len(chunks)} 块分词完成")


def build_vector(chunks: list[dict]) -> bool:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("未安装 sentence-transformers/chromadb，跳过向量索引（将使用 BM25 检索）")
        return False

    print(f"加载嵌入模型 {EMBED_MODEL}（首次运行会下载模型，可能较慢）...")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    collection = client.get_or_create_collection(name="campus")

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "title": c["title"],
            "category": c["category"],
            "rel_path": c["rel_path"],
            "heading": c.get("heading", ""),
            "doc_id": c["doc_id"],
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"向量索引：{len(chunks)} 块已入库 ChromaDB")
    return True


def main() -> None:
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    if not chunks:
        print("chunks.json 为空，请先运行 parse.py 与 chunk.py")
        return

    build_bm25(chunks)
    vector_ok = build_vector(chunks)

    (DATA_DIR / "index_ready.json").write_text(
        json.dumps({"vector": vector_ok, "bm25": True, "chunks": len(chunks)}, indent=2),
        encoding="utf-8",
    )
    print("索引构建完成")


if __name__ == "__main__":
    main()
