# -*- coding: utf-8 -*-
"""DeepSeek 客户端层——OpenAI 兼容 SDK + 环境变量 Key + 响应缓存 + JSON 解析。

固化四件影响正确性/复现性的事:

1. Key 只读 os.environ["DEEPSEEK_API_KEY"](绝不进代码/config/仓库/压缩包);
   可选 project/.env 文件兜底加载——十余行手写解析器, 不引入 python-dotenv
   (轻依赖纪律)。.env 已进 .gitignore, 仓库只有 .env.example 空键名模板。

2. 思考模式显式关闭: extra_body={"thinking": {"type": "disabled"}}——端点默认
   enabled, 不关则 temperature 无效 + reasoning token 抬高延迟费用。
   temperature=0; seed 端点不支持、不发送(该事实由本文件常量+日志落盘,
   复现性真正兜底是缓存); frequency/presence_penalty 已废弃不使用。

3. 响应缓存 = 复现 + 省钱: 键 = sha256(model+thinking+temperature+
   max_tokens+response_format+messages 全量规范序列化)——漏任一参数会串缓存。
   缓存 JSON 落 project/cache/llm_<hash16>.json(含原始响应/usage/
   延迟), 与 emb_*.npy 同目录约定: 可再生、不进压缩包。命中即零二次请求,
   同 prompt 重跑逐字节一致。

4. 结构化输出: response_format={"type":"json_object"}(要求 prompt 含 "json"
   字样——generate 的协议模板天然满足); 官方注明偶发空 content, 由调用方
   (generate 的重试一次)覆盖。解析三级降级: 直接 loads → 剥代码围栏 → 括号
   配平扫描, 降级路径逐调用落日志。

延迟与成本核算: 每次调用记录墙钟延迟与 usage 细项(prompt_cache_hit_tokens /
prompt_cache_miss_tokens / reasoning_tokens——DeepSeek 私有字段, SDK 模型
未必声明, 一律走 getattr+model_extra 防御读取, 冒烟 7.4 实测核对)。

用法: 由 src/generate.py 调用; 缺 Key 时抛 RuntimeError 并附设置指引。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT

# flash 谷时定价 USD/1M tokens(峰时[工作日北京时间 09-12/14-18]×2)
PRICING_PER_MTOK = {"input_miss": 0.15, "input_hit": 0.003, "output": 0.60}

# seed 不在端点参数表——不发送, 复现性由响应缓存兜底
UNSUPPORTED_PARAMS = ("seed", "frequency_penalty", "presence_penalty")


# ---------- .env 加载(手写十余行, 不引 python-dotenv) ----------

def load_env_file(path: Path) -> dict[str, str]:
    """极简 .env 解析: KEY=VALUE 逐行, #注释/空行跳过, 引号剥壳。

    只做 KEY=VALUE 一件事——不支持 export/多行/变量展开, 保持零依赖零歧义。
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_api_key(llm_cfg: dict) -> str:
    """os.environ 优先(已设则 .env 不覆盖), project/.env 兜底;缺失给设置指引。"""
    env_name = llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    if os.environ.get(env_name):
        return os.environ[env_name]
    from_file = load_env_file(PROJECT_ROOT / ".env").get(env_name, "")
    if from_file:
        return from_file
    raise RuntimeError(
        f"缺少 {env_name}: 请设置环境变量, 或在 project/.env 写一行 {env_name}=sk-…"
        f"(.env 已被 .gitignore 排除; 模板见 project/.env.example)。"
        f"无 Key 环境下 run_all 的生成阶段会整段跳过, 检索层不受影响。")


# ---------- JSON 三级解析(降级链) ----------

def extract_json(text: str) -> tuple[dict | None, str]:
    """content → (解析出的 dict 或 None, 路径标签)。

    路径: "direct"(直接 loads) / "fence"(剥 ```json 围栏) / "balanced"(首个
    括号配平的顶层对象) / ""(三级全败)。降级路径由调用方落日志。
    """
    if text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj, "direct"
        except (json.JSONDecodeError, ValueError):
            pass
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict):
                    return obj, "fence"
            except (json.JSONDecodeError, ValueError):
                pass
        depth, start, in_str, esc = 0, None, False, False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj, "balanced"
                    except (json.JSONDecodeError, ValueError):
                        start = None   # 该顶层块非法, 继续扫下一个
    return None, ""


# ---------- 客户端 ----------

class LLMClient:
    """DeepSeek(OpenAI 兼容)聊天客户端: 缓存优先, 网络为兜底。

    chat(messages)      -> 原始层: {"content", "model", "finish_reason", "usage",
                             "latency_seconds", "cache_hit", "request_hash"}
    chat_json(messages) -> 解析层: chat 结果 + {"parsed", "parse_path"}
    两层都纯函数式记录, generate 只消费不解析协议细节。
    """

    def __init__(self, llm_cfg: dict, cache_dir: Path = PROJECT_ROOT / "cache"):
        self.model = llm_cfg.get("model", "deepseek-flash")
        self.base_url = llm_cfg.get("base_url", "https://api.deepseek.com")
        self.thinking = llm_cfg.get("thinking", "disabled")
        self.temperature = llm_cfg.get("temperature", 0)
        self.max_tokens = llm_cfg.get("max_tokens", 1024)
        self.api_key = get_api_key(llm_cfg)
        self.cache_dir = Path(cache_dir)
        self._client = None   # OpenAI() 延迟实例化: 缓存全命中时零网络零 SDK 依赖

    # ---- 请求规范化与缓存键 ----

    def _request_spec(self, messages: list[dict]) -> dict:
        """参与缓存键的全量请求参数(漏参数 = 换模型/开关思考串缓存)。"""
        return {"model": self.model,
                "thinking": {"type": self.thinking},
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": messages}

    def _cache_key(self, messages: list[dict]) -> str:
        blob = json.dumps(self._request_spec(messages), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"llm_{key[:16]}.json"

    @staticmethod
    def _usage_dict(u) -> dict:
        """usage → 平 dict;DeepSeek 私有字段缓存命中细项走 model_extra 防御读取。"""
        if u is None:
            return {}
        extra = dict(getattr(u, "model_extra", None) or {})
        d = {"prompt_tokens": getattr(u, "prompt_tokens", None),
             "completion_tokens": getattr(u, "completion_tokens", None),
             "total_tokens": getattr(u, "total_tokens", None)}
        for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            d[k] = getattr(u, k, extra.get(k))
        ctd = getattr(u, "completion_tokens_details", None)
        d["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", None) if ctd \
            else extra.get("completion_tokens_details", {}).get("reasoning_tokens")
        return {k: v for k, v in d.items() if v is not None}

    # ---- 网络层(测试桩替换点: 子类覆写即可离线全链演练) ----

    def _raw_chat(self, messages: list[dict]) -> dict:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        t0 = time.time()
        # 网络瞬断/限流退避重试(90 题连跑的稳健性;最终失败原样上抛, 由缓存重跑兜底)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": self.thinking}},
                )
                break
            except Exception as e:   # openai.APIError 系: 记录后退避重试
                last_exc = e
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
        else:
            raise last_exc
        latency = round(time.time() - t0, 3)
        msg = resp.choices[0].message
        return {"content": msg.content,
                "model": getattr(resp, "model", None),
                "finish_reason": resp.choices[0].finish_reason,
                "id": getattr(resp, "id", None),
                "usage": self._usage_dict(getattr(resp, "usage", None)),
                "latency_seconds": latency}

    def chat(self, messages: list[dict]) -> dict:
        """缓存优先的一次调用;返回 dict 原样含原始响应、usage、延迟与命中标记。"""
        key = self._cache_key(messages)
        path = self._cache_path(key)
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                return {**cached["response"], "cache_hit": True,
                        "request_hash": key}
            except (OSError, json.JSONDecodeError, KeyError):
                pass   # 缓存损坏视同未命中, 网络重取后覆写
        out = self._raw_chat(messages)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"request_hash": key,
             "created_at": datetime.now().isoformat(timespec="seconds"),
             "response": out}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**out, "cache_hit": False, "request_hash": key}

    def chat_json(self, messages: list[dict]) -> dict:
        r = self.chat(messages)
        parsed, path = extract_json(r.get("content") or "")
        return {**r, "parsed": parsed, "parse_path": path}


def cost_usd(usage: dict) -> dict:
    """按 usage 细项估算单次/汇总费用(USD)。缺命中细项时按全未命中口径(上界)。"""
    p = usage.get("prompt_tokens") or 0
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        hit, miss = 0, p
    else:
        hit = hit or 0
        miss = miss if miss is not None else max(p - hit, 0)
    comp = usage.get("completion_tokens") or 0
    offpeak = (hit * PRICING_PER_MTOK["input_hit"]
               + miss * PRICING_PER_MTOK["input_miss"]
               + comp * PRICING_PER_MTOK["output"]) / 1e6
    # 8 位小数: 单次调用费用 ~1e-5 USD 量级, 6 位会累积求和误差
    return {"hit": hit, "miss": miss, "completion": comp,
            "usd_offpeak": round(offpeak, 8), "usd_peak": round(offpeak * 2, 8)}
