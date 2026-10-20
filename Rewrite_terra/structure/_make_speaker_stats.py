import re
from collections import Counter

PATH = r'E:/dev/NLP/Rewrite_terra/structure/seen08000.dat.txt'
OUT = r'E:/dev/NLP/Rewrite_terra/structure/speaker_labels.md'

VERIFIED = {
    '瑚太朗＿必要': ('瑚太朗', '用户已在游戏确认'),
    '男＿ルイス': ('男人', '用户已在游戏确认（剧情前期不知其名，汉化显示泛称）'),
    '路易斯': ('路易斯', '用户已在游戏确认（原名直显）'),
}

with open(PATH, encoding='utf-8') as f:
    lines = f.read().splitlines()

cd_re = re.compile(r'^(\d+):\s*\S+\s+CD_NAME "(.*)"')
str_re = re.compile(r'^(\d+):?\[?(\d+)\]? \'(.*)\'$')

# str_list section: lines 17 .. 20917 (namae_list header at 20918)
str_defs = {}  # label -> [(lineno, str_idx), ...]
for i in range(16, 20917):
    m = re.match(r"^\[(\d+)\] '(.*)'$", lines[i])
    if m:
        str_defs.setdefault(m.group(2), []).append((i + 1, int(m.group(1))))

name_uses = []  # (lineno, label)
for i, ln in enumerate(lines):
    if 'CD_NAME ' in ln:
        m = re.search(r'CD_NAME "(.*)"', ln)
        if m:
            name_uses.append((i + 1, m.group(1)))

counter = Counter(lb for _, lb in name_uses)
occ = {}
for no, lb in name_uses:
    occ.setdefault(lb, []).append(no)

def fmt_lines(xs, k=3):
    if len(xs) <= k:
        return ', '.join(str(x) for x in xs)
    return ', '.join(str(x) for x in xs[:k]) + f' … {xs[-1]}'

dist = Counter()
for n in counter.values():
    if n >= 100: dist['>=100'] += 1
    elif n >= 10: dist['10-99'] += 1
    elif n >= 2: dist['2-9'] += 1
    else: dist['=1'] += 1

sfx = Counter()
for lb in counter:
    parts = lb.split('＿')
    if len(parts) > 1:
        sfx['＿'.join(parts[1:])] += 1

out = []
out.append('# 说话人原始标签统计（说话人归一化表底稿）\n')
out.append('> 来源：`structure/seen08000.dat.txt` 的 `CD_NAME` 行（说话人切换标记，共 '
           f'{len(name_uses)} 处、{len(counter)} 个不同标签）。')
out.append('> "str_list 行"= 该标签在字符串表中的定义位置；"CD_NAME 首现/末现行"= 说话人切换指令在反汇编中的位置，')
out.append('> 前后几行即 `CD_TEXT read_flag=N ; "台词"`，可对照该处台词进游戏核对实际显示名。\n')
out.append('## 已由游戏确认的映射（3 条）\n')
out.append('| 原始标签 | 游戏显示名 | 依据位置（seen08000.dat.txt） |')
out.append('|---|---|---|')
out.append('| 瑚太朗＿必要 | **瑚太朗** | 行 39108（`CD_NAME`），下行 39115 即台词 read_flag=1「…好疼…」 |')
out.append('| 男＿ルイス | **男人** | 行 39135（`CD_NAME`），行 39142 即台词 read_flag=3「你没事吧？」 |')
lou_first = min(occ.get('路易斯', [0]))
out.append(f'| 路易斯 | **路易斯** | 行 {lou_first}（首次 `CD_NAME` 路易斯） |')
out.append('')
out.append('推测的显示规则（待更多对照验证）：`＿xxx` 后缀在显示时剥离（`瑚太朗＿必要`→`瑚太朗`）；')
out.append('但**不是所有标签都是"前缀+后缀"的机械关系**——`男＿ルイス` 显示为泛称"男人"（剧情前期瑚太朗还不知道对方姓名），')
out.append('这类"未知姓名期变体"必须逐个人工登记，不能只靠去后缀。\n')
out.append('## 标签分布概览\n')
out.append(f'- 不同标签总数：**{len(counter)}**；CD_NAME 总次数：**{len(name_uses)}**')
out.append(f"- 使用次数分布：≥100 次 {dist['>=100']} 个；10–99 次 {dist['10-99']} 个；2–9 次 {dist['2-9']} 个；仅 1 次 {dist['=1']} 个")
once = sum(1 for lb in counter if counter[lb] == 1)
out.append(f"- 仅出现 1 次的标签（多为一次性路人，可直接归并为「旁白/路人」类）：{once} 个")
if sfx:
    out.append('- 带 `＿` 后缀的标签变体后缀统计（去后缀前需逐个核对）：'
               + '、'.join(f'`＿{k}`×{v}' for k, v in sfx.most_common()))
out.append('')
out.append('## 全部标签（按出现次数降序）\n')
out.append('| # | 原始标签 | 次数 | str_list 行 | CD_NAME 首现 | CD_NAME 末现 | 游戏显示名（待对照） | 备注 |')
out.append('|---|---|---|---|---|---|---|---|')
rank = 0
for lb, n in counter.most_common():
    rank += 1
    defs = str_defs.get(lb, [])
    d_str = fmt_lines([f"{no}(str{idx})" for no, idx in defs], 2) if defs else '—'
    verified, note = VERIFIED.get(lb, ('', ''))
    if not verified:
        note = '一次性路人？' if n == 1 else ''
    out.append(f"| {rank} | `{lb}` | {n} | {d_str} | {occ[lb][0]} | {occ[lb][-1]} | {verified or '？'} | {note} |")

out.append('')
out.append('## 用法说明\n')
out.append('1. 打开游戏（SiglusEngineCHS.exe）读到对应台词处，记录右侧说话人框的实际显示文字，回填"游戏显示名"列。')
out.append('2. 优先核对高频标签（表头前 ~30 行）和高频后缀变体；仅 1 次的路人标签可以抽查。')
out.append('3. 回填完成后，把该表转成 `speaker_map.tsv`（原始标签 → 显示名），供 `lines.jsonl` 的 `speaker_norm` 字段使用。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')

print(f"labels={len(counter)} uses={len(name_uses)} dist={dict(dist)}")
print(f"written: {OUT}")
print("top10:", counter.most_common(10))
