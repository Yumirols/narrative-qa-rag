import re

FILES = [
    r"H:/key/rewrite/SiglusEngineCHS.exe",
    r"H:/key/rewrite/SiglusEngine.original.exe",
    r"H:/key/rewrite/SiglusEngine.exe",
    r"H:/key/rewrite/Gameexe.dat",
    r"H:/key/rewrite/gameexe.chs",
    r"H:/key/rewrite/scene.chs",
]
PATTERNS = ["男＿ルイス", "瑚太朗＿必要", "男人", "路易斯", "＿必要", "？＿瑚太朗", "瑚太朗", "河村"]

def find_all(data: bytes, pat: bytes):
    out, start = [], 0
    while True:
        i = data.find(pat, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1
        if len(out) > 8:
            break
    return out

for fp in FILES:
    with open(fp, "rb") as f:
        data = f.read()
    hits = []
    for s in PATTERNS:
        for enc in ("utf-16-le", "gbk"):
            b = s.encode(enc)
            offs = find_all(data, b)
            if offs:
                hits.append(f"{s}[{enc}]x{len(offs)}@{offs[0]:#x}" + (f"..{offs[-1]:#x}" if len(offs) > 1 else ""))
    print(f"{fp.split('/')[-1]}: size={len(data)}")
    for h in hits:
        print("   ", h)
    if not hits:
        print("    (no hits)")
    print()
