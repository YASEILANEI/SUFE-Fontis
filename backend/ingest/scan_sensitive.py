"""敏感信息扫描：比赛禁止上传含个人隐私、账号密码等敏感信息的材料。

严格命中（入库时跳过该块）：手机号 / 身份证号 / 银行卡号 / 明文密码与账号值。
弱命中（仅告警）：出现"密码/账号"等描述词但无实际值。
"""
from __future__ import annotations

import re

_STRICT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b1[3-9]\d{9}\b"), "手机号"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "身份证号"),
    (re.compile(r"\b(?:\d[ -]?){16,19}\b"), "银行卡号"),
    (re.compile(r"(?:密码|口令|account|password)\s*[:：=]\s*[\w@#$%^&*!]{4,}", re.IGNORECASE), "明文密码/账号值"),
]

_WEAK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:账号|账户|密码|口令|登录名|身份证|手机号|银行卡)", re.IGNORECASE), "含账号/密码类描述词"),
]


def find_sensitive(text: str) -> list[str]:
    """返回命中的敏感模式名列表。空列表表示可安全入库。"""
    hits: list[str] = []
    for pattern, name in _STRICT_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


def has_weak_marker(text: str) -> list[str]:
    return [name for pattern, name in _WEAK_PATTERNS if pattern.search(text)]


def filter_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """按敏感命中过滤分块。返回 (保留的块, 被过滤的块)。"""
    kept: list[dict] = []
    dropped: list[dict] = []
    for c in chunks:
        hits = find_sensitive(c["text"])
        if hits:
            c["sensitive"] = hits
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped
