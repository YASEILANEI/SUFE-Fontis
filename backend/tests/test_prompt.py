"""build_user_prompt：无片段时给一般性说明 + 免责声明，有片段时带编号标注。"""
from app.qa import build_user_prompt


def _chunk(i):
    return {
        "id": f"doc#{i}",
        "doc_id": "学工/测试",
        "title": f"测试文件{i}",
        "heading": "第三章",
        "text": f"片段{i}内容",
    }


def test_prompt_without_chunks_mentions_missing_and_disclaimer():
    prompt = build_user_prompt([], "校医院怎么挂号")
    assert "未检索到相关官方资料" in prompt
    assert "普遍性常识问题" in prompt
    assert "以学校正式文件" in prompt


def test_prompt_with_chunks_lists_sources():
    prompt = build_user_prompt([_chunk(1)], "奖学金条件")
    assert "官方资料片段" in prompt
    assert "[1]" in prompt and "片段1内容" in prompt
    assert "以片段为准" in prompt
    assert "学生最新问题：奖学金条件" in prompt
