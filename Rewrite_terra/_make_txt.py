import csv
import re

CJK = re.compile(r'[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]')


def unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s):
            n = s[i + 1]
            if n == '\\':
                out.append('\\')
            elif n == 'n':
                out.append('\n')
            elif n == 'r':
                out.append('\r')
            elif n == 't':
                out.append('\t')
            else:
                out.append(c)
                i -= 1  # \uXXXX etc: keep literal
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


rows = []
with open(r'E:/dev/NLP/seen08000.dat.csv', encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        rows.append((int(row['index']), int(row['kind']), unescape(row['replacement'])))

speaker = None
lines = []
dialogue_chars = 0
n_dialogue = n_speaker = n_other = 0
for idx, kind, text in rows:
    if kind == 2:
        speaker = text
        n_speaker += 1
    elif kind == 1:
        n_dialogue += 1
        dialogue_chars += len(text.replace('\n', '').replace('\r', ''))
        lines.append(f"【{speaker or '？'}】{text}")
    else:
        n_other += 1
        if CJK.search(text):  # drop pure-ASCII resource names like cgm_sz12
            lines.append(text)

with open(r'E:/dev/NLP/Terra_汉化文本.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"total_strings={len(rows)} dialogue={n_dialogue} speaker_names={n_speaker} other={n_other}")
print(f"dialogue_chars={dialogue_chars}")
