# -*- coding: utf-8 -*-
"""
第 8 讲 · 神经语言模型:RNN / LSTM(纯 numpy 手写)
====================================================

RNN:每个时间步复用同一组权重(参数共享),把历史压进隐状态 h:
    h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b_h)        x_t 为 one-hot 经 embedding 的稠密向量
    y_t = Why·h_t + b_y,   P(下一字符) = softmax(y_t)
训练 = BPTT(沿时间反向传播):损失从最后一步倒着传,每步先算输出层梯度 dy,
再经 cell 回传到 dx / dh_{t-1},参数梯度逐步累加。

RNN 的死穴:Whh 反复相乘 → 梯度消失/爆炸。
LSTM 的解法:细胞状态 c_t 走「加法」高速通道,门控逐元素调控:
    i_t = σ(...),  f_t = σ(...),  o_t = σ(...),  g_t = tanh(...)   (输入/遗忘/输出门 + 候选)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      <- 加法!梯度可沿 f 长距离流动
    h_t = o_t ⊙ tanh(c_t)
实现技巧:把四组门拼成一次矩阵乘 a = Wx·x + Wh·h + b 再按 [i|f|o|g] 切开。

反向传播关键公式(LSTM 单步,前向已存中间量):
    do = dh ⊙ tanh(c);  dc = dh ⊙ o ⊙ (1-tanh²c) + dc_next
    df = dc ⊙ c_prev;   di = dc ⊙ g;    dg = dc ⊙ i;   dc_prev = dc ⊙ f
    门激活的导数照 sigmoid/tanh 求导,拼回 da 后一次乘出 dWx、dWh。

文件结构:
  1. RNNCell / LSTMCell:手写前向 + 反向,并用数值梯度检查验证
  2. 字符级语言模型:小语料训练,对比 RNN 与 LSTM 的损失下降并采样
"""
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------- Cell 1:朴素 RNN ----------
class RNNCell:
    """h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b)"""

    def __init__(self, D, H, rng):
        self.Wxh = rng.normal(0, 0.1, (H, D))
        self.Whh = rng.normal(0, 0.1, (H, H))
        self.b = np.zeros(H)
        self.zero_grad()

    def params(self):
        return [self.Wxh, self.Whh, self.b]

    def grads(self):
        return [self.dWxh, self.dWhh, self.db]

    def zero_grad(self):
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.db = np.zeros_like(self.b)

    def forward(self, x, h_prev):
        h_next = np.tanh(self.Wxh @ x + self.Whh @ h_prev + self.b)
        return h_next, (x, h_prev, h_next)

    def backward(self, dh, cache):
        """输入:损失对 h_t 的总梯度;返回:对 x_t 与 h_{t-1} 的梯度。"""
        x, h_prev, h = cache
        da = dh * (1.0 - h * h)                    # tanh' = 1 - tanh²
        self.dWxh += np.outer(da, x)
        self.dWhh += np.outer(da, h_prev)
        self.db += da
        return self.Wxh.T @ da, self.Whh.T @ da


# ---------- Cell 2:LSTM ----------
class LSTMCell:
    """四门拼接:a = Wx·x + Wh·h + b ∈ R^{4H},按 [i | f | o | g] 切开。"""

    def __init__(self, D, H, rng):
        self.H = H
        self.Wx = rng.normal(0, 0.1, (4 * H, D))
        self.Wh = rng.normal(0, 0.1, (4 * H, H))
        self.b = np.zeros(4 * H)
        self.zero_grad()

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def grads(self):
        return [self.dWx, self.dWh, self.db]

    def zero_grad(self):
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

    def forward(self, x, h_prev, c_prev):
        a = self.Wx @ x + self.Wh @ h_prev + self.b
        H = self.H
        i = sigmoid(a[:H])                  # 输入门
        f = sigmoid(a[H:2 * H])             # 遗忘门
        o = sigmoid(a[2 * H:3 * H])         # 输出门
        g = np.tanh(a[3 * H:])              # 候选内容
        c_next = f * c_prev + i * g
        tanh_c = np.tanh(c_next)
        h_next = o * tanh_c
        return h_next, c_next, (x, h_prev, c_prev, i, f, o, g, tanh_c)

    def backward(self, dh_next, dc_next, cache):
        x, h_prev, c_prev, i, f, o, g, tanh_c = cache
        do = dh_next * tanh_c
        dc = dh_next * o * (1.0 - tanh_c ** 2) + dc_next   # 梯度两路来源:经 h 与直接经 c
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f
        da = np.concatenate([                              # 过门激活的导数
            di * i * (1.0 - i),                            # sigmoid'
            df * f * (1.0 - f),
            do * o * (1.0 - o),
            dg * (1.0 - g ** 2),                           # tanh'
        ])
        self.dWx += np.outer(da, x)
        self.dWh += np.outer(da, h_prev)
        self.db += da
        return self.Wx.T @ da, self.Wh.T @ da, dc_prev


# ---------- 数值梯度检查 ----------
def _num_grad(f, arr, eps=1e-5):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        ix = it.multi_index
        old = arr[ix]
        arr[ix] = old + eps
        lp = f()
        arr[ix] = old - eps
        lm = f()
        arr[ix] = old
        g[ix] = (lp - lm) / (2 * eps)
    return g


def grad_check_cell(kind, D=3, H=4, seed=0):
    """用线性化的「损失」L = Σr⊙h (+ Σr2⊙c),解析/数值梯度逐一对比。"""
    rng = np.random.default_rng(seed)
    cell = LSTMCell(D, H, rng) if kind == "lstm" else RNNCell(D, H, rng)
    x = rng.normal(size=D)
    h0 = rng.normal(size=H)
    c0 = rng.normal(size=H) if kind == "lstm" else None
    r1 = rng.normal(size=H)
    r2 = rng.normal(size=H) if kind == "lstm" else None

    def loss():
        if kind == "lstm":
            h, c, _ = cell.forward(x, h0, c0)
            return float(r1 @ h + r2 @ c)
        h, _ = cell.forward(x, h0)
        return float(r1 @ h)

    cell.zero_grad()                                       # 解析梯度
    pairs = []                                             # (被求导数组, 解析梯度, 名称)
    if kind == "lstm":
        _, _, cache = cell.forward(x, h0, c0)
        dx, dh0, dc0 = cell.backward(r1, r2, cache)
        pairs += [(x, dx, "dx"), (h0, dh0, "dh0"), (c0, dc0, "dc0")]
    else:
        _, cache = cell.forward(x, h0)
        dx, dh0 = cell.backward(r1, cache)
        pairs += [(x, dx, "dx"), (h0, dh0, "dh0")]
    pnames = ["dWx", "dWh", "db"] if kind == "lstm" else ["dWxh", "dWhh", "db"]
    pairs += list(zip(cell.params(), cell.grads(), pnames))

    worst = 0.0
    for arr, g_ana, name in pairs:
        g_num = _num_grad(loss, arr)
        err = np.max(np.abs(g_ana - g_num) / np.maximum(np.abs(g_ana) + np.abs(g_num), 1e-8))
        worst = max(worst, float(err))
    return worst


# ---------- 字符级语言模型 ----------
class CharLM:
    """embedding(D 维) + cell + 输出投影。整个片段按时间展开做 BPTT。"""

    def __init__(self, vocab_size, D=8, H=64, kind="lstm", seed=1):
        rng = np.random.default_rng(seed)
        self.kind, self.H, self.V = kind, H, vocab_size
        self.E = rng.normal(0, 0.1, (vocab_size, D))
        self.cell = LSTMCell(D, H, rng) if kind == "lstm" else RNNCell(D, H, rng)
        self.Why = rng.normal(0, 0.1, (vocab_size, H))
        self.by = np.zeros(vocab_size)
        self.zero_grad()

    def zero_grad(self):
        self.cell.zero_grad()
        self.dE = np.zeros_like(self.E)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def params(self):
        return [self.E, *self.cell.params(), self.Why, self.by]

    def grads(self):
        return [self.dE, *self.cell.grads(), self.dWhy, self.dby]

    def loss_and_grads(self, ids_in, ids_out):
        """前向整个片段,平均交叉熵损失 + BPTT 累加所有参数梯度。"""
        self.zero_grad()
        T = len(ids_in)
        hs = [np.zeros(self.H)]
        cs = [np.zeros(self.H)] if self.kind == "lstm" else None
        caches = []
        logits = np.zeros((T, self.V))
        for t in range(T):
            x = self.E[ids_in[t]]
            if self.kind == "lstm":
                h, c, cache = self.cell.forward(x, hs[-1], cs[-1])
                cs.append(c)
            else:
                h, cache = self.cell.forward(x, hs[-1])
            hs.append(h)
            caches.append(cache)
            logits[t] = self.Why @ h + self.by

        logits -= logits.max(axis=1, keepdims=True)        # softmax(数值稳定)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(probs[np.arange(T), ids_out] + 1e-12)))

        dh_next = np.zeros(self.H)
        dc_next = np.zeros(self.H) if self.kind == "lstm" else None
        for t in reversed(range(T)):
            dy = probs[t].copy()
            dy[ids_out[t]] -= 1.0
            dy /= T                                        # 平均损失 → 梯度同除 T
            self.dWhy += np.outer(dy, hs[t + 1])
            self.dby += dy
            dh = self.Why.T @ dy + dh_next
            if self.kind == "lstm":
                dx, dh_prev, dc_prev = self.cell.backward(dh, dc_next, caches[t])
                dc_next = dc_prev
            else:
                dx, dh_prev = self.cell.backward(dh, caches[t])
            dh_next = dh_prev
            self.dE[ids_in[t]] += dx
        return loss


def train_char_lm(text, kind, iters=450, seq_len=20, D=8, H=64, lr=0.1, seed=1):
    """Adagrad + 全局梯度裁剪,随机切窗口训练。返回 (模型, 最终损失, 采样函数)。"""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = np.array([stoi[ch] for ch in text])

    model = CharLM(len(chars), D, H, kind, seed)
    m_cache = [np.zeros_like(p) for p in model.params()]

    def clip_grads(max_norm=5.0):
        total = np.sqrt(sum(float((g * g).sum()) for g in model.grads()))
        if total > max_norm:
            s = max_norm / (total + 1e-8)
            for g in model.grads():
                g *= s

    def sample_fn(n=60, temp=0.8, seed=7):
        rng = np.random.default_rng(seed)
        h = np.zeros(H)
        c = np.zeros(H) if kind == "lstm" else None
        idx = int(rng.integers(0, len(chars)))         # 从词表随机起点开始接龙
        out = []
        for _ in range(n):
            x = model.E[idx]
            if kind == "lstm":
                h, c, _ = model.cell.forward(x, h, c)
            else:
                h, _ = model.cell.forward(x, h)
            z = (model.Why @ h + model.by) / temp
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            idx = int(rng.choice(len(chars), p=p))
            out.append(itos[idx])
        return "".join(out)

    rng = np.random.default_rng(seed)
    loss = None
    for it in range(1, iters + 1):
        s = int(rng.integers(0, len(data) - seq_len - 1))
        chunk = data[s:s + seq_len + 1]
        loss = model.loss_and_grads(list(chunk[:-1]), list(chunk[1:]))
        clip_grads()
        for p, g, m in zip(model.params(), model.grads(), m_cache):   # Adagrad
            m += g * g
            p -= lr * g / np.sqrt(m + 1e-8)
        if it == 1 or it % 150 == 0:
            print(f"    [{kind:4}] iter {it:4d}  loss = {loss:.3f}")
            print(f"            采样: {sample_fn()!r}")
    return model, loss, sample_fn


if __name__ == "__main__":
    print("── [1] 数值梯度检查(验证手推的反向传播)──")
    for kind in ["rnn", "lstm"]:
        err = grad_check_cell(kind)
        print(f"  {kind.upper():4} 最大相对误差 = {err:.2e}  ->  {'PASS ✔' if err < 1e-6 else 'FAIL ✘'}")
        assert err < 1e-6, f"{kind} 梯度检查未通过"

    TEXT = "the cat sat on the mat . the dog ate the bone . the bird sang a song . " * 6
    print(f"── [2] 字符级语言模型(语料 {len(TEXT)} 字符,RNN vs LSTM 同任务对比)──")
    _, loss_rnn, sample_rnn = train_char_lm(TEXT, kind="rnn")
    _, loss_lstm, sample_lstm = train_char_lm(TEXT, kind="lstm")

    print("── [3] 训练后的生成效果对比 ──")
    print(f"  RNN  最终损失 {loss_rnn:.3f}")
    print(f"       采样: {sample_rnn(n=90)!r}")
    print(f"  LSTM 最终损失 {loss_lstm:.3f}")
    print(f"       采样: {sample_lstm(n=90)!r}")
    print("\n观察:LSTM 损失降得更快更低、采样更接近训练文本 —— 门控缓解了梯度消失")
