"""Locate retrieved chunks inside parsed document text."""
from __future__ import annotations

import re


def _line_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        stripped = content.strip()
        if stripped:
            start = offset + len(content) - len(content.lstrip())
            end = offset + len(content.rstrip())
            spans.append((stripped, start, end))
        offset += len(line)
    return spans


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _compact_with_offsets(text: str) -> tuple[str, list[int]]:
    chars, offsets = [], []
    for i, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            offsets.append(i)
    return "".join(chars), offsets


def _find_span(full_text: str, target: str) -> tuple[int, int] | None:
    full_lines = _line_spans(full_text)
    target_lines = [line for line, _, _ in _line_spans(target)]
    if target_lines and len(target_lines) <= len(full_lines):
        for i in range(len(full_lines) - len(target_lines) + 1):
            if all(full_lines[i + j][0] == line for j, line in enumerate(target_lines)):
                return full_lines[i][1], full_lines[i + len(target_lines) - 1][2]

    compact_full, offsets = _compact_with_offsets(full_text)
    compact_target = _compact(target)
    if not compact_target:
        return None
    start = compact_full.find(compact_target)
    if start < 0:
        return None
    end = start + len(compact_target) - 1
    return offsets[start], offsets[end] + 1


def compute_citations(
    full_text: str,
    chunks_by_id: dict[str, dict],
    cite_ids: list[str],
) -> list[dict]:
    """Return sorted, merged source ranges in ``full_text``."""
    ranges = []
    for chunk_id in cite_ids:
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        span = _find_span(full_text, chunk.get("text", ""))
        if span is None:
            continue
        start, end = span
        ranges.append({
            "chunk_id": chunk_id,
            "heading": chunk.get("heading", ""),
            "start": start,
            "end": end,
        })
    ranges.sort(key=lambda item: (item["start"], item["end"]))
    merged = []
    for item in ranges:
        if merged and item["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        else:
            merged.append(item)
    return merged
