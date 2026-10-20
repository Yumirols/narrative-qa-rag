"""Parse seen08000.dat.txt disassembly -> lines.jsonl, scenes.jsonl, choices.json

输入: structure/seen08000.dat.txt (siglus-ssu -x --disam 产物)
      structure/speaker_map.tsv (#NAMAE 导出的 说话人归一化映射)
输出: data/lines.jsonl   行级真值结构(每条台词一行)
      data/scenes.jsonl  三级边界信号切分后的 chunk
      data/choices.json  选择肢节点
v2: 选择肢真伪标注(has_branch) + 分支状态机切分(branch_ids) + 终点检测(terminal/ending)
    旧版输出备份在 data_v1/ 供对比
"""
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(r"E:/dev/NLP/Rewrite_terra")
DAT = BASE / "structure" / "seen08000.dat.txt"
SPEAKER_MAP = BASE / "structure" / "speaker_map.tsv"
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

SCENE = "seen08000"
SRC_LINE_GAP = 40   # 源行号前跳超过该值视为章节边界; 后跳/原地视为边界
MAX_CHARS = 500     # chunk 目标字符上限(消融时改此参数重跑)
MIN_CHARS = 40      # 低于该值的碎片 chunk 并入前一个


def unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "\\":
                out.append("\\")
            elif n == "n":
                out.append("\n")
            elif n == "r":
                out.append("\r")
            elif n == "t":
                out.append("\t")
            else:
                out.append(c)
                i -= 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ---- speaker map ----
speaker_map = {}
with open(SPEAKER_MAP, encoding="utf-8") as f:
    for ln in f.readlines()[1:]:
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0]:
            speaker_map[parts[0]] = parts[1]

# ---- pass 1: parse disassembly into line records ----
re_label = re.compile(r"^([0-9A-F]+): <([LZ][0-9, LZ]+)>$")
re_nl = re.compile(r": CD_NL (\d+)$")
re_name = re.compile(r'CD_NAME "(.*)"$')
re_text = re.compile(r'CD_TEXT read_flag=(\d+) ; "(.*)"$')
re_bgm = re.compile(r'expr=global\.bgm\.play\("([^"]+)"\)')
re_farcall = re.compile(r'expr=global\.farcall\("([^"]+)"(.*)\)')
re_selready = re.compile(r"expr=global\.selbtn_ready\((.*)\)")
re_selstart = re.compile(r"expr=global\.selbtn_start\(\)")
re_cmd_readflag = re.compile(r"read_flag=(\d+)")
# v2 分支状态机
re_gate = re.compile(r"CD_GOTO_FALSE (L\d+) -> [0-9A-F]+ ; cond=\(global\.c\[810\] == (\d+)\)")
re_goto = re.compile(r": CD_GOTO L(\d+) -> [0-9A-F]+")
re_terminal = re.compile(r"syscom\.return_to_menu|global\.end_close")
re_writer = re.compile(r"expr=global\.c\[810\] = global\.selbtn")
re_selbtn_cmd = re.compile(r"expr=global\.selbtn\(\d+, (.*)\)$")

lines_out = []
choices = []
label = None
src_line = None
speaker_raw = None
bgm = bg = None
last = None  # previous emitted record
deltas = []
# 分支状态机状态
branch_ids = None        # 当前可达分支集合(set[int]) / None=无分支上下文
label_srcs = {}          # CD_GOTO 目标 label -> 跳转来源分支集合
dead = False             # GOTO/终点之后到下一标签前是死代码, 不更新状态
pending_break = False    # 强制下条文本开新 segment
cur_gates = []           # 当前写入点收集到的分支门 N 值
last_writer_choice = None  # 当前写入点对应的 choices 下标
region = None            # (choice下标, 区域起始seq); 写入点到下一写入点之间
regions = []             # 全部区域 (choice下标, start_seq, end_seq)
selbtn_pending = False   # 刚收集到 selbtn( 一次性选择, 等它的赋值行来接管区域


def close_writer():
    """写入点结束: 把该写入点收集到的分支门挂到对应 choice"""
    if last_writer_choice is not None and cur_gates and last_writer_choice < len(choices):
        choices[last_writer_choice]["has_branch"] = True
        choices[last_writer_choice]["gate_values"] = sorted(set(cur_gates))


for ln in open(DAT, encoding="utf-8").read().splitlines():
    m = re_label.match(ln)
    if m:
        items = re.findall(r"([LZ])(\d+)", m.group(2))
        labs = [int(n) for t, n in items if t == "L"]
        if labs:
            label = labs[0]
        dead = False
        arrived = set()
        for L in labs:
            arrived |= label_srcs.get(L, set())
        if arrived:  # 汇合点: 跳转来源分支并入 fall-through 分支
            branch_ids = (branch_ids or set()) | arrived
        continue
    m = re_nl.search(ln)
    if m:
        src_line = int(m.group(1))
        continue
    if "CD_NAME " in ln:
        m = re_name.search(ln)
        if m:
            speaker_raw = m.group(1)
        continue
    if "expr=global.bgm.play" in ln:
        m = re_bgm.search(ln)
        if m:
            bgm = m.group(1)
        continue
    if "expr=global.farcall" in ln:
        m = re_farcall.search(ln)
        if m and m.group(1) == "sys20_adv00":
            qs = re.findall(r'"([^"]*)"', m.group(2))
            if len(qs) >= 2:
                bg = qs[-1]
        continue
    if "expr=global.selbtn_ready" in ln:
        m = re_selready.search(ln)
        if m:
            opts = re.findall(r'"([^"]*)"', m.group(1))
            choices.append({"kind": "selbtn_start", "options": opts, "src_line": src_line,
                            "label": label, "addr": ln.split(":")[0]})
        continue
    m = re_selbtn_cmd.search(ln)
    if m:
        # 一次性选择: selbtn 直接返回选中下标, 不走 ready/start 两段式 (战斗等即时选择)
        opts = re.findall(r'"([^"]*)"', m.group(1))
        rf = re_cmd_readflag.search(ln)
        choices.append({"kind": "selbtn", "options": opts, "src_line": src_line,
                        "label": label, "addr": ln.split(":")[0],
                        "rfid": int(rf.group(1)) if rf else None})
        selbtn_pending = True
        continue
    if re_writer.search(ln):
        # 新写入点: 上一个写入点的分支期结束
        close_writer()
        if region is not None:
            regions.append((region[0], region[1], len(lines_out)))
        cur_gates = []
        branch_ids = None
        label_srcs = {}
        dead = False  # 不清 pending_break: 终点后的强制分段必须保留
        new_choice = (len(choices) - 1) if ("selbtn_start" in ln or selbtn_pending) else None
        selbtn_pending = False
        region = (new_choice, len(lines_out))
        last_writer_choice = new_choice
        # 不 continue: selbtn_start 赋值行仍需走下方 rfid 提取
    if "expr=global.selbtn_start" in ln:
        m = re_selstart.search(ln)
        if m and choices and "rfid" not in choices[-1]:
            rf = re_cmd_readflag.search(ln)
            choices[-1]["rfid"] = int(rf.group(1)) if rf else None
        continue
    m = re_gate.search(ln)
    if m:
        # 分支门: c[810]==N 为真的内容顺延其后
        branch_ids = {int(m.group(2))}
        cur_gates.append(int(m.group(2)))
        dead = False
        pending_break = True
        continue
    m = re_goto.search(ln)
    if m:
        if not dead and branch_ids:  # 记录汇合点的跳转来源; 死代码里的 GOTO 不算
            label_srcs.setdefault(int(m.group(1)), set()).update(branch_ids)
        dead = True
        continue
    if re_terminal.search(ln):
        # 终点: return_to_menu / end_close 不返回, 其后死代码的跳转全部作废
        if lines_out:
            lines_out[-1]["terminal"] = ("return_to_menu" if "return_to_menu" in ln
                                         else "end_close")
        branch_ids = None
        label_srcs = {}
        dead = True
        pending_break = True
        continue
    m = re_text.search(ln)
    if m:
        rfid = int(m.group(1))
        text = unescape(m.group(2))
        is_dialogue = ("「" in text) or ("『" in text)
        speaker_norm = speaker_map.get(speaker_raw, speaker_raw) if is_dialogue and speaker_raw else (
            speaker_map.get(speaker_raw, speaker_raw) if speaker_raw else "旁白")
        if not is_dialogue:
            speaker_norm = "旁白"  # 叙述挂旁白, 原始标签保留在 speaker_raw 备查
        cur_branch = sorted(branch_ids) if branch_ids else None
        rec = {"rfid": rfid, "text": text,
               "speaker_raw": speaker_raw, "speaker_norm": speaker_norm,
               "is_dialogue": is_dialogue, "src_line": src_line,
               "label": label, "bgm": bgm, "bg": bg, "branch": cur_branch}
        rec["seq"] = len(lines_out)
        if last is not None:
            d = (src_line or 0) - (last["src_line"] or 0)
            deltas.append(d)
            rec["new_segment"] = bool(label != last["label"] or bgm != last["bgm"]
                                      or bg != last["bg"] or d <= 0 or d > SRC_LINE_GAP
                                      or pending_break or cur_branch != last["branch"])
        else:
            rec["new_segment"] = True
        pending_break = False
        lines_out.append(rec)
        last = rec
close_writer()
if region is not None:
    regions.append((region[0], region[1], len(lines_out)))
for c in choices:
    c.setdefault("has_branch", False)

unmapped = sorted({r["speaker_raw"] for r in lines_out
                   if r["is_dialogue"] and r["speaker_raw"] and r["speaker_raw"] not in speaker_map})
n_dialogue = sum(1 for r in lines_out if r["is_dialogue"])
n_empty = sum(1 for r in lines_out if not r["text"])
term_seqs = [r["seq"] for r in lines_out if "terminal" in r]
route_end_seq = max(term_seqs) if term_seqs else None

# ---- pass 2: segment -> chunks (空文本行保留在 lines.jsonl, 但不进入 chunk) ----
segments = []
for r in lines_out:
    if not r["text"]:
        continue
    if r["new_segment"] or not segments:
        segments.append([])
    segments[-1].append(r)

chunks = []
for seg in segments:
    cur, cur_chars = [], 0
    for r in seg:
        t = len(r["text"])
        if cur and cur_chars + t > MAX_CHARS:
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append(r)
        cur_chars += t
    if cur:
        chunks.append(cur)


def _br(rs):
    return rs[0]["branch"]


# 碎片 chunk 并入前一个(开头的并入后一个); 跨分支边界不并(v2)
merged = []
for c in chunks:
    if (merged and sum(len(r["text"]) for r in c) < MIN_CHARS
            and _br(merged[-1]) == _br(c)):
        merged[-1].extend(c)
    else:
        merged.append(c)
if (merged and len(merged) > 1 and sum(len(r["text"]) for r in merged[0]) < MIN_CHARS
        and _br(merged[0]) == _br(merged[1])):
    merged[1][:0] = merged[0]
    merged.pop(0)
chunks = merged

def chunk_text(rs):
    parts = []
    for r in rs:
        parts.append(f"【{r['speaker_norm']}】{r['text']}" if r["is_dialogue"] else r["text"])
    return "\n".join(parts)

warn = []
chunk_recs = []
for i, rs in enumerate(chunks):
    if len({tuple(r["branch"] or ()) for r in rs}) > 1:  # 断言1: chunk 内分支一致
        warn.append(f"chunk#{i} 混合分支: {sorted({tuple(r['branch'] or ()) for r in rs})}")
    rfids = [r["rfid"] for r in rs]
    rec = {
        "chunk_id": f"{SCENE}_c{i + 1:03d}",
        "scene": SCENE,
        "line_start": rs[0]["seq"], "line_end": rs[-1]["seq"],
        "rfid_start": min(rfids), "rfid_end": max(rfids),
        "label": rs[0]["label"],
        "bgm": rs[0]["bgm"], "bg": rs[0]["bg"],
        "branch_ids": rs[0]["branch"],
        "speakers": sorted({r["speaker_norm"] for r in rs if r["speaker_norm"] != "旁白"}),
        "n_lines": len(rs),
        "n_chars": sum(len(r["text"]) for r in rs),
        "text": chunk_text(rs),
    }
    tseq = next((r["seq"] for r in rs if "terminal" in r), None)
    if tseq is not None:
        rec["terminal"] = True
        rec["ending"] = "route_end" if tseq == route_end_seq else "bad_end"
    chunk_recs.append(rec)
for j, cr in enumerate(chunk_recs):  # 断言2: 终点后必须换分支/换chunk
    if cr.get("terminal") and j + 1 < len(chunk_recs) \
            and chunk_recs[j + 1]["branch_ids"] == cr["branch_ids"]:
        warn.append(f"{cr['chunk_id']} 终点后未换分支")

# choices -> attach chunk ids (选择肢 rfid 是独立事件号, 取 rfid_start <= rfid 的最后一个 chunk)
for seq_c, c in enumerate(choices, 1):
    c["seq"] = seq_c
    c["chunk_id"] = None
    best = None
    for cr in chunk_recs:
        if c["rfid"] is not None and cr["rfid_start"] <= c["rfid"]:
            best = cr
        else:
            break
    if best is not None:
        c["chunk_id"] = best["chunk_id"]
        best["has_choice"] = True
        best.setdefault("choices", []).append(c["seq"])
for cr in chunk_recs:
    cr.setdefault("has_choice", False)
    cr.setdefault("choices", [])

# v2: 真分叉的选项级标注 (分支 -> 独占/共通 chunk 与结局)
for c in choices:
    if not c["has_branch"]:
        continue
    if c.get("options"):  # 断言3: 分支门 N 必须落在选项下标范围内
        bad = [n for n in c["gate_values"] if n >= len(c["options"])]
        if bad:
            warn.append(f"choice#{c['seq']} 分支门越界: {bad} (选项数{len(c['options'])})")
    reg = next((r for r in regions if r[0] == c["seq"] - 1), None)
    region_chunks = ([cr for cr in chunk_recs
                      if reg[1] <= cr["line_start"] < reg[2]] if reg else [])
    blist = []
    for n in c["gate_values"]:
        cs = [cr for cr in region_chunks if cr["branch_ids"] and n in cr["branch_ids"]]
        outcome = "continue"
        for cr in cs:
            if cr.get("terminal"):
                outcome = cr["ending"]
                break
        blist.append({"idx": n, "chunk_ids": [x["chunk_id"] for x in cs], "outcome": outcome})
    c["branches"] = blist

with open(OUT / "lines.jsonl", "w", encoding="utf-8") as f:
    for r in lines_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "scenes.jsonl", "w", encoding="utf-8") as f:
    for c in chunk_recs:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(OUT / "choices.json", "w", encoding="utf-8") as f:
    json.dump({"scene": SCENE, "count": len(choices), "choices": choices},
              f, ensure_ascii=False, indent=2)

sizes = [c["n_chars"] for c in chunk_recs]
hist = Counter("neg" if d < 0 else "0" if d == 0 else "1-5" if d <= 5 else "6-40" if d <= 40 else ">40" for d in deltas)
n_bad = sum(1 for cr in chunk_recs if cr.get("ending") == "bad_end")
n_route = sum(1 for cr in chunk_recs if cr.get("ending") == "route_end")
print(f"lines={len(lines_out)} (dialogue={n_dialogue}, narration={len(lines_out) - n_dialogue}, empty_text={n_empty})")
print(f"unmapped_dialogue_speakers={unmapped}")
print(f"choices={len(choices)} (has_branch={sum(1 for c in choices if c['has_branch'])}, "
      f"with rfid={sum(1 for c in choices if c.get('rfid'))}, "
      f"in_chunk={sum(1 for c in choices if c['chunk_id'])})")
print(f"terminal chunks: bad_end={n_bad}, route_end={n_route}")
print(f"状态机断言: {'全部通过' if not warn else warn}")
print(f"segments={len(segments)} chunks={len(chunk_recs)} "
      f"chars min/avg/max={min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}")
print(f"speakers in use: {len({r['speaker_norm'] for r in lines_out})}")
print(f"src_line deltas: {dict(hist)}")
