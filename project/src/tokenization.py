# -*- coding: utf-8 -*-
"""检索字段的归一化与双分词臂(知识框架 §3.2 / §3.3)。

- normalize():      全角字母数字符号 → 半角,仅用于检索字段;
                    证据展示永远用原文 text(半角化 ２５岁→25岁 属于改写)。
- jieba_tokenize(): 词典分词臂,加载 data/user_dict.txt(说话人真值+专名)。
- bigram_tokenize():字符 bigram 臂,绕过词典——原理点①(分词/子词影响下游)的对照。

两臂通过 get_tokenizer() 以同一接口注入 BM25 索引。
注意:jieba 的自定义词典只能增量加载、无法卸载。run_all.py 先用默认词典
切样例句(切分"前"),再调 jieba_tokenize(..., userdict=...) 完成加载(切分"后"),
此后进程内所有 jieba 分词都带自定义词典。
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_jieba_dict_loaded = False


def normalize(text: str) -> str:
    """全角(FF01–FF5E)→半角平移,全角空格(U+3000)→半角空格。CJK 标点不动。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _has_word(tok: str) -> bool:
    """过滤纯标点/空白 token(「」。【】,等)。"""
    return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in tok)


def jieba_tokenize(text: str, userdict: str | Path | None = None) -> list[str]:
    """词典分词臂。首次传入 userdict 时加载(进程内只加载一次)。"""
    global _jieba_dict_loaded
    import jieba

    if userdict is not None and not _jieba_dict_loaded:
        jieba.load_userdict(str(userdict))
        _jieba_dict_loaded = True
    return [t for t in jieba.lcut(normalize(text)) if _has_word(t)]


def bigram_tokenize(text: str) -> list[str]:
    """字符 bigram 臂:汉字取相邻 bigram(孤立单字保留),ASCII 词整存。"""
    text = normalize(text)
    out: list[str] = []
    ascii_buf: list[str] = []
    cjk_buf: list[str] = []

    def flush_cjk() -> None:
        if not cjk_buf:
            return
        if len(cjk_buf) == 1:
            out.append(cjk_buf[0])
        else:
            out.extend(cjk_buf[i] + cjk_buf[i + 1] for i in range(len(cjk_buf) - 1))
        cjk_buf.clear()

    for ch in text:
        if ch.isascii() and ch.isalnum():
            flush_cjk()
            ascii_buf.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            if ascii_buf:
                out.append("".join(ascii_buf))
                ascii_buf.clear()
            cjk_buf.append(ch)
        else:
            if ascii_buf:
                out.append("".join(ascii_buf))
                ascii_buf.clear()
            flush_cjk()
    if ascii_buf:
        out.append("".join(ascii_buf))
    flush_cjk()
    return out


def get_tokenizer(name: str, userdict: str | Path | None = None):
    """按名字返回分词函数。

    jieba  臂在首次调用时加载自定义词典;
    raw    臂 = jieba 默认词典,不触发加载——原理点①的"无词典"对照。
    注意 raw 与 jieba 两臂不可在同一进程内混用(jieba 词典加载后无法卸载):
    raw 须先建,或各自在独立进程中运行(CLI 每次调用天然是新进程)。
    """
    if name == "jieba":
        return lambda t: jieba_tokenize(t, userdict)
    if name == "raw":
        return jieba_tokenize
    if name == "bigram":
        return bigram_tokenize
    raise ValueError(f"未知分词臂: {name}(可选 jieba / raw / bigram)")
