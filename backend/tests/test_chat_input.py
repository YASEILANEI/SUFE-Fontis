"""/api/chat 输入校验：问题/分类长度上限、历史裁剪（防滥用防护）。"""
from app.main import (
    MAX_CATEGORY,
    MAX_HISTORY,
    MAX_HISTORY_CONTENT,
    MAX_QUESTION,
    normalize_history,
    validate_chat_input,
)


def test_validate_accepts_normal_input():
    assert validate_chat_input("差旅住宿费报销的限额是多少？", "学工") is None


def test_validate_rejects_empty_question():
    assert validate_chat_input("", "") == "问题不能为空"


def test_validate_rejects_overlong_question():
    assert (
        validate_chat_input("长" * (MAX_QUESTION + 1), "")
        == f"问题过长（最多 {MAX_QUESTION} 字）"
    )


def test_validate_accepts_exactly_max_question():
    assert validate_chat_input("长" * MAX_QUESTION, "") is None


def test_validate_rejects_overlong_category():
    assert validate_chat_input("问题", "长" * (MAX_CATEGORY + 1)) == "分类参数过长"


def test_normalize_history_keeps_only_last_max():
    history = [{"role": "user", "content": f"问{i}"} for i in range(30)]
    out = normalize_history(history)
    assert len(out) == MAX_HISTORY
    assert out[-1]["content"] == "问29"
    assert out[0]["content"] == "问10"


def test_normalize_history_truncates_long_content():
    out = normalize_history([{"role": "user", "content": "长" * (MAX_HISTORY_CONTENT + 100)}])
    assert len(out) == 1
    assert len(out[0]["content"]) == MAX_HISTORY_CONTENT


def test_normalize_history_filters_roles_empty_and_dupes():
    history = [
        {"role": "system", "content": "系统消息"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "回答"},
        {"role": "user", "content": "追问"},
        {"role": "tool", "content": "工具消息"},
    ]
    out = normalize_history(history)
    assert [m["content"] for m in out] == ["回答", "追问"]


def test_normalize_history_handles_none_and_bad_items():
    assert normalize_history(None) == []
    out = normalize_history([{"role": "user", "content": 123}, "not-a-dict", {"role": "user", "content": None}])
    assert out == [{"role": "user", "content": "123"}]
