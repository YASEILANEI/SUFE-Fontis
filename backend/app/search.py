"""检索器：向量检索（可用时）+ BM25 关键词检索，结果合并去重。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"


class Retriever:
    def __init__(self, k: int = 6) -> None:
        self.k = k
        self.chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}
        self.status = json.loads((DATA_DIR / "index_ready.json").read_text(encoding="utf-8"))

        self._vector = None
        if self.status.get("vector"):
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer

                self._vector = {
                    "model": SentenceTransformer(os.getenv("EMBED_MODEL", "bge-m3")),
                    "collection": chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
                    .get_collection(name="campus"),
                }
            except Exception as exc:  # 模型或库加载失败则退回 BM25
                print(f"向量检索初始化失败，降级为 BM25：{exc}")
                self._vector = None

        tokens = json.loads((DATA_DIR / "bm25_tokens.json").read_text(encoding="utf-8"))
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([t["tokens"] for t in tokens])
        self._bm25_ids = [t["id"] for t in tokens]

    def _vector_top(self, query: str, category: str | None = None) -> list[str]:
        emb = self._vector["model"].encode([query])
        kwargs = {"where": {"category": category}} if category else {}
        res = self._vector["collection"].query(
            query_embeddings=emb.tolist(), n_results=self.k, **kwargs
        )
        return [cid for cid in res["ids"][0] if cid in self.by_id]

    def _bm25_top(self, query: str, category: str | None = None) -> list[str]:
        scores = self._bm25.get_scores(jieba.lcut(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
        for i in ranked:
            cid = self._bm25_ids[i]
            if scores[i] <= 0:
                break
            if category and self.by_id[cid]["category"] != category:
                continue
            out.append(cid)
            if len(out) >= self.k * 2:
                break
        return out

    def retrieve(self, query: str, k: int | None = None, category: str | None = None) -> list[dict]:
        k = k or self.k
        combined: dict[str, int] = {}
        if self._vector:
            for rank, cid in enumerate(self._vector_top(query, category)):
                combined[cid] = combined.get(cid, 0) + (k - rank) * 2
        for rank, cid in enumerate(self._bm25_top(query, category)):
            combined[cid] = combined.get(cid, 0) + (k - rank)

        top = sorted(combined, key=combined.get, reverse=True)[:k]
        return [dict(self.by_id[cid], score=combined[cid]) for cid in top]
