# -*- coding: utf-8 -*-
"""渲染 w2v 训练 loss 曲线图(数据只来自 runs/ 日志,禁止手绘)。

读取 runs/<ts>/w2v_train_log.json(必需;run_all 落盘)与
w2v_divergence_reference.json(可选;修复前发散参考,一次性诊断产物),
输出同目录 w2v_loss_curve.png:
  - 面板1: 现行代码的逐 epoch 平均损失,附随机初始化理论起点参考线
    ((1+neg_k)·ln2 ≈ 3.47,W_out 零初始化 → 全部 σ≈0.5)——判读曲线的锚点;
  - 面板2(有参考产物时): 修复前发散曲线对照,用于核对实现细节/复现性。

用法:
  python src/plot_w2v_loss.py                    # 默认渲染含 w2v 日志的最新 run
  python src/plot_w2v_loss.py runs/20261023-120000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT

LN2 = 0.6931


def _baseline(neg_k: int) -> float:
    """训练前的理论平均损失: 所有 sigmoid=0.5 时 -log(0.5)·(1+neg_k)。"""
    return (1 + neg_k) * LN2


def _tuck_title(fig, title, axes, gap: float = 0.036) -> None:
    """把总标题收到坐标区上方:tight_layout 为 suptitle 预留的空白按 rect 与文字包围盒
    双重扣除,两行标题下会留下近 1/6 图高的空档;这里按实际包围盒重设位置。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    axes_top = max(a.get_tightbbox(renderer).transformed(inv).y1 for a in axes)
    height = title.get_window_extent(renderer).transformed(inv).height
    title.set_position((0.5, axes_top + gap + height))


def _annotate(ax, curve: list[float]) -> None:
    ax.annotate(f"{curve[0]:.2f}", (1, curve[0]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)
    # 末点贴底,标注放上方空白处,避开底轴与末尾刻度标签
    ax.annotate(f"{curve[-1]:.2f}", (len(curve), curve[-1]),
                xytext=(-4, 8), textcoords="offset points", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染 w2v loss 曲线(数据源: runs 日志)")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="runs/<ts> 目录;缺省=含 w2v_train_log.json 的最新 run")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = PROJECT_ROOT / args.run_dir
    else:
        cands = sorted(p for p in (PROJECT_ROOT / "runs").iterdir()
                       if p.is_dir() and (p / "w2v_train_log.json").exists())
        if not cands:
            sys.exit("runs/ 下没有含 w2v_train_log.json 的 run——先跑 run_all.py")
        run_dir = cands[-1]

    log = json.loads((run_dir / "w2v_train_log.json").read_text(encoding="utf-8"))
    curve, params = log["loss_curve"], log["params"]
    ref_path = run_dir / "w2v_divergence_reference.json"
    ref = (json.loads(ref_path.read_text(encoding="utf-8"))
           if ref_path.exists() else None)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2 if ref else 1, figsize=(6.4 * (2 if ref else 1), 4.2),
                             dpi=150)
    axes = list(axes) if ref else [axes]

    ax = axes[0]
    base = _baseline(params["neg_k"])
    ax.axhline(base, ls="--", lw=1, color="gray", label=f"随机初始化理论起点 ≈{base:.2f}")
    ax.plot(range(1, len(curve) + 1), curve, "o-", lw=1.6, ms=4, label="修复后(现行代码)")
    _annotate(ax, curve)
    ax.set_xlabel("epoch")
    ax.set_ylabel("平均每对损失")
    ax.set_title("现行代码: 单调下降,正常收敛")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if ref:
        rc = ref["loss_curve"]
        ax2 = axes[1]
        ax2.plot(range(1, len(rc) + 1), rc, "s-", lw=1.6, ms=4, color="tab:red",
                 label="修复前(梯度误用更新后的 v)")
        ax2.axhline(base, ls="--", lw=1, color="gray")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("平均每对损失")
        ax2.set_title("修复前对照: 饱和震荡,权重发散")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)

    p = (f"dim={params['dim']}  window={params['window']}  neg_k={params['neg_k']}  "
         f"epochs={params['epochs']}  lr={params['lr']}  seed={params['seed']}  "
         f"vocab={log['vocab']:,}  训练={log['train_seconds']}s")
    st = fig.suptitle(f"w2v_avg skip-gram 训练损失 — {run_dir.name}\n{p}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _tuck_title(fig, st, axes)
    out = run_dir / "w2v_loss_curve.png"
    fig.savefig(out)
    print(f"已渲染: {out}")
    print(f"  loss {curve[0]} → {curve[-1]} (min {min(curve)}),"
          f" 数据源: {(run_dir / 'w2v_train_log.json').name}"
          + (" + w2v_divergence_reference.json" if ref else ""))


if __name__ == "__main__":
    main()
