"""LLM 相关性重排：按 LLM 输出编号重排，失败/空/无效编号回退原候选。"""
import pytest

from app import qa


def _message(content):
    return type("Msg", (), {"content": content})


def _choice(content):
    return type("Choice", (), {"message": _message(content)})


def _resp(content):
    return type("Resp", (), {"choices": [_choice(content)]})


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeClient:
    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def _chunk(i):
    return {"id": f"doc#{i}", "doc_id": "doc", "heading": f"第{i}章", "text": f"第{i}章 内容内容内容内容内容"}


def _candidates(n):
    return [_chunk(i) for i in range(1, n + 1)]


def test_rerank_reorders_by_llm_output(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("3 1 2")]))
    result = qa.rerank_chunks("奖学金条件", _candidates(5), top_k=3)
    assert [c["heading"] for c in result] == ["第3章", "第1章", "第2章"]


def test_rerank_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("5 4 3 2 1")]))
    result = qa.rerank_chunks("问题", _candidates(6), top_k=3)
    assert [c["heading"] for c in result] == ["第5章", "第4章", "第3章"]


def test_rerank_falls_back_on_empty_output(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("")]))
    result = qa.rerank_chunks("无关问题", _candidates(8), top_k=6)
    assert [c["heading"] for c in result] == [f"第{i}章" for i in range(1, 7)]


def test_rerank_ignores_invalid_ids(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("2 abc 99 1")]))
    result = qa.rerank_chunks("问题", _candidates(4), top_k=3)
    assert [c["heading"] for c in result] == ["第2章", "第1章"]


def test_rerank_falls_back_on_llm_exception(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([RuntimeError("llm down")]))
    result = qa.rerank_chunks("问题", _candidates(10), top_k=6)
    assert len(result) == 6


def test_rerank_skips_llm_when_few_candidates(monkeypatch):
    fake = FakeClient([])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    result = qa.rerank_chunks("问题", _candidates(3), top_k=6)
    assert len(result) == 3  # 候选本就 ≤top_k，不调用 LLM


def test_rerank_truncates_long_text():
    long_chunk = {"id": "d", "doc_id": "doc", "heading": "长", "text": "甲" * 500}
    assert qa._RERANK_TEXT_MAX == 220  # 截断上限
    numbered = f"[1] {long_chunk['text'].strip()[:qa._RERANK_TEXT_MAX]}"
    assert len(numbered) == 4 + 220  # "[1] "（4 字符）+ 220 字
