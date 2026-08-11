"""LLM 查询改写：成功使用改写词，失败/空输出回退原 query，结果缓存。"""
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
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeClient:
    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


@pytest.fixture(autouse=True)
def clear_cache():
    qa._rewrite_cache.clear()
    yield
    qa._rewrite_cache.clear()


def test_rewrite_uses_llm_output(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("课程考核不合格 毕业要求")]))
    assert qa.rewrite_query("挂科了还能不能顺利毕业") == "课程考核不合格 毕业要求"


def test_rewrite_falls_back_on_empty_output(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([_resp("")]))
    assert qa.rewrite_query("贫困生补助怎么申请") == "贫困生补助怎么申请 家庭经济困难学生 资助"


def test_rewrite_falls_back_on_llm_exception(monkeypatch):
    monkeypatch.setattr(qa, "get_client", lambda: FakeClient([RuntimeError("llm down")]))
    assert qa.rewrite_query("导师可以换吗") == "导师可以换吗"


def test_rewrite_caches_success(monkeypatch):
    fake = FakeClient([_resp("毕业要求 课程考核")])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    assert qa.rewrite_query("挂科影响毕业吗") == "毕业要求 课程考核"
    assert qa.rewrite_query("挂科影响毕业吗") == "毕业要求 课程考核"
    assert fake.chat.completions.calls == 1


def test_rewrite_does_not_cache_failure(monkeypatch):
    fake = FakeClient([RuntimeError("llm down"), _resp("家庭经济困难学生 资助")])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    assert qa.rewrite_query("贫困生补助") == "贫困生补助 家庭经济困难学生 资助"  # 异常→词典兜底，不缓存
    assert qa.rewrite_query("贫困生补助") == "家庭经济困难学生 资助"  # 恢复后重新调用 LLM
    assert fake.chat.completions.calls == 2


def test_rewrite_retries_when_unchanged(monkeypatch):
    # 第一次原样返回（仍含口语词），触发重试，第二次改写成功
    fake = FakeClient([_resp("贫困生补助怎么申请"), _resp("家庭经济困难学生 资助")])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    assert qa.rewrite_query("贫困生补助怎么申请") == "家庭经济困难学生 资助"
    assert fake.chat.completions.calls == 2


def test_rewrite_gives_up_after_two_unchanged(monkeypatch):
    # 连续两次未改写，最终回退到词典扩展，且不缓存
    fake = FakeClient([_resp("贫困生补助怎么申请"), _resp("贫困生补助怎么申请")])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    assert qa.rewrite_query("贫困生补助怎么申请") == "贫困生补助怎么申请 家庭经济困难学生 资助"
    assert fake.chat.completions.calls == 2


def test_rewrite_retries_when_too_short(monkeypatch):
    # 第一次只输出'校园卡'（过短，丢了补办/挂失语义），触发重试
    fake = FakeClient([_resp("校园卡"), _resp("校园卡补办 挂失 流程")])
    monkeypatch.setattr(qa, "get_client", lambda: fake)
    assert qa.rewrite_query("校园卡丢了应该怎么办") == "校园卡补办 挂失 流程"
    assert fake.chat.completions.calls == 2
