from app.citations import compute_citations


def test_compute_citations_maps_chunk_to_original_offsets():
    text = "第一章\n\n学生应当遵守校规。\n第二章\n奖学金按综合成绩评定。"
    chunks = {
        "doc#1": {
            "id": "doc#1",
            "text": "学生应当遵守校规。",
            "heading": "第一章",
        }
    }

    citations = compute_citations(text, chunks, ["doc#1"])

    assert citations == [{
        "chunk_id": "doc#1",
        "heading": "第一章",
        "start": text.index("学生应当遵守校规。"),
        "end": text.index("学生应当遵守校规。") + len("学生应当遵守校规。"),
    }]


def test_compute_citations_handles_whitespace_and_unknown_chunks():
    text = "办理流程：\n第一步  提交材料\n第二步：等待审核"
    chunks = {
        "doc#1": {"id": "doc#1", "text": "第一步\n提交材料", "heading": "流程"},
        "missing": {"id": "missing", "text": "不存在的内容", "heading": ""},
    }

    citations = compute_citations(text, chunks, ["doc#1", "missing"])

    assert len(citations) == 1
    assert text[citations[0]["start"]:citations[0]["end"]].replace(" ", "") == "第一步提交材料"


def test_compute_citations_merges_overlapping_ranges():
    text = "甲乙丙丁戊"
    chunks = {
        "a": {"id": "a", "text": "乙丙", "heading": "A"},
        "b": {"id": "b", "text": "丙丁", "heading": "B"},
    }

    citations = compute_citations(text, chunks, ["a", "b"])

    assert citations == [{"chunk_id": "a", "heading": "A", "start": 1, "end": 4}]
