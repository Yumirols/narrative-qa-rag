# 语料落位目录

`configs/main.yaml.example` 的 `paths.scenes` / `paths.choices` 默认指向本目录：

```yaml
paths:
  scenes:  data/corpus/scenes.jsonl
  choices: data/corpus/choices.json
```

## 放什么

按 `README.md`（项目根）§1.2 从网盘“数据集”文件夹取得以下三份文件，直接放在此处即可运行，
无需修改配置：

| 文件 | 内容 | 大小（参考） |
| --- | --- | --- |
| `lines.jsonl` | 11,033 行剧本记录（台词 4,358 / 旁白 6,675） | ~4.5 MB |
| `scenes.jsonl` | 483 个 chunk（三级边界信号切分） | ~1.5 MB |
| `choices.json` | 52 个选择肢节点 | ~30 KB |

`speaker_map.tsv` 只在重建语料（§3.4）时用到，不必放在这里。

## 校验

放好后比对 sha256，应与项目根 `README.md` §3.6 的记录一致：

```bash
sha256sum data/corpus/lines.jsonl data/corpus/scenes.jsonl data/corpus/choices.json
```

## 说明

- 语料是受版权保护的原始文本，**不进 git、不进提交包**（`.gitignore` 已排除本目录下的
  `*.json` / `*.jsonl` / `*.tsv`，仅本说明文件入库）。
- 语料放在别处（例如项目外目录）也行，只需改配置的 `paths.scenes` 与 `paths.choices`；
  相对路径以项目根为基准（`src/config.py:resolve_paths`），也可写绝对路径。
- 本目录为空时运行 `run_all.py` 会在读文件前提示缺失的具体路径，不会抛原始异常。
