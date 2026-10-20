# -*- coding: utf-8 -*-
"""
第 2 讲 · 文本清洗与正则表达式(Text Cleaning & Regex)
========================================================

正则表达式是 NLP 数据预处理的「手术刀」。本文件对应课程第 2 讲的三类典型任务:

  1. 信息抽取 —— 从原始文本提取邮箱 / 手机号 / URL / 日期等结构化字段
  2. 文本清洗 —— 去 HTML 标签、全角转半角、合并空白
  3. 数据脱敏 —— 对手机号 / 邮箱打码,保护隐私

正则速查
--------
字符类   [abc] 任取其一   [^abc] 取反   \\d 数字   \\w 字母数字下划线   \\s 空白
量词     * 0次以上   + 1次以上   ? 0或1次   {m,n} m到n次
分组     (...) 捕获   (?:...) 不捕获   (?P<name>...) 命名分组
逻辑     | 或
贪婪     默认贪婪(尽量多匹配),加 ? 变非贪婪 .*?
常用API  re.findall 找全部 / re.search 找第一个 / re.sub 替换 / re.match 从头匹配
"""
import re
import sys

if sys.platform == "win32":  # Windows 控制台默认 GBK,统一为 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 先定义、多复用、可单测 —— 工程上的好习惯
EMAIL_RE = re.compile(r"(?P<name>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PHONE_RE = re.compile(r"1[3-9]\d{9}")          # 大陆手机号:1 开头,第二位 3-9,共 11 位
URL_RE   = re.compile(r"https?://[^\s,，。”\"<>]+")
DATE_RE  = re.compile(r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?")


# ---------- 1. 信息抽取 ----------
def extract(text: str) -> dict:
    """抽取四类结构化字段,返回 {字段: [命中列表]}。

    注意:邮箱模式带命名分组(供脱敏复用),findall 会返回分组元组,
    因此这里用 finditer().group(0) 拿完整匹配。
    """
    return {
        "email": [m.group(0) for m in EMAIL_RE.finditer(text)],
        "phone": PHONE_RE.findall(text),
        "url":   URL_RE.findall(text),
        "date":  DATE_RE.findall(text),
    }


# ---------- 2. 文本清洗 ----------
TAG_RE = re.compile(r"<[^>]+>")           # 普通标签,如 <p>、<br/>
SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
SPACE_RE = re.compile(r"\s+")             # 连续空白(含换行)
FULL2HALF = str.maketrans({c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)})
FULL2HALF[0x3000] = 0x20                  # 全角空格 U+3000 -> 半角空格


def normalize(text: str) -> str:
    """全角转半角 + 合并空白。中英文混排清洗的基本功。"""
    return SPACE_RE.sub(" ", text.translate(FULL2HALF)).strip()


def strip_html(html: str) -> str:
    """先整块去掉 <script>/<style>(防止残留 JS 代码),再去普通标签。"""
    return normalize(TAG_RE.sub(" ", SCRIPT_RE.sub(" ", html)))


# ---------- 3. 数据脱敏 ----------
def mask_phone(text: str) -> str:
    """手机号保留前 3 后 4:138****8000。re.sub 支持函数替换。"""
    return PHONE_RE.sub(lambda m: m.group()[:3] + "****" + m.group()[-4:], text)


def mask_email(text: str) -> str:
    """邮箱用户名只留首字符:z***@qq.com。命名分组让替换逻辑更可读。"""
    return EMAIL_RE.sub(lambda m: m["name"][:1] + "***@" + m["domain"], text)


def clean_pipeline(raw: str) -> str:
    """把上面串成一条流水线:去标签 -> 规范化 -> 脱敏。"""
    return mask_email(mask_phone(strip_html(raw)))


if __name__ == "__main__":
    sample = (
        "<div style='x'>联系我:zhang.san@mail.neu.edu.cn,电话 13812345678。</div>"
        "<p>项目主页 https://nlp.neu.edu.cn/course,截止 2026-10-15。</p>"
        "<script>alert('垃圾代码')</script>"
    )

    print("── [1] 信息抽取 ──")
    for k, v in extract(sample).items():
        print(f"  {k:6}: {v}")

    print("── [2] 文本清洗 ──")
    print("  strip_html :", strip_html(sample))

    print("── [3] 数据脱敏 ──")
    print("  mask_phone :", mask_phone(sample))
    print("  mask_email :", mask_email(sample))

    print("── [4] 全流程流水线 ──")
    print("  clean_pipeline:", clean_pipeline(sample))

    # ---- 自检 ----
    r = extract(sample)
    assert r["email"] == ["zhang.san@mail.neu.edu.cn"]
    assert r["phone"] == ["13812345678"]
    assert "nlp.neu.edu.cn/course" in r["url"][0]
    assert r["date"] == ["2026-10-15"]
    cleaned = clean_pipeline(sample)
    assert "<" not in cleaned and "alert" not in cleaned
    assert "138****5678" in cleaned and "z***@mail.neu.edu.cn" in cleaned
    print("\n所有自检通过 ✔")
