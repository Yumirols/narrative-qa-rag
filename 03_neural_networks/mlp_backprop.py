# -*- coding: utf-8 -*-
"""
第 6 讲 · 神经网络基础:感知机 -> 多层感知机(MLP)-> 反向传播
================================================================

1. 感知机  y = step(w·x + b)
   只能解「线性可分」问题。Minsky 1969 指出 XOR 不可分 -> 第一次 AI 寒冬。

2. MLP = 线性层 + 非线性激活堆叠。万能逼近定理:单个隐层足够宽即可逼近任意连续函数。

3. 反向传播 = 链式法则的系统性应用(前向存中间量,反向倒着乘):
      前向:z[l] = W[l]·a[l-1] + b[l],   a[l] = g(z[l])
      输出层(softmax + 交叉熵):delta[L] = ŷ - y        <- 极其漂亮的结果
      隐层:sigmoid 导数 = a(1-a),delta[l] = (W[l+1]^T·delta[l+1]) ⊙ a[l](1-a[l])
      参数:dW[l] = delta[l]·a[l-1]^T,   db[l] = Σ delta[l]

4. 梯度检查:数值梯度 (L(θ+ε) - L(θ-ε)) / 2ε 验证手推公式 —— 写网络必会
   的调试神技。⚠ 检查时用 float64、ε≈1e-5,训练时可转 float32 提速。
"""
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 1. 感知机 ----------
class Perceptron:
    """经典感知机学习算法:错分类驱动更新 w += lr·t·x(t ∈ {+1,-1})。"""

    def __init__(self, dim, lr=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.1, dim)
        self.b = 0.0
        self.lr = lr

    def fit(self, X, y, epochs=50):
        t = np.where(y == 1, 1.0, -1.0)
        for ep in range(epochs):
            errors = 0
            for xi, ti in zip(X, t):
                if ti * (self.w @ xi + self.b) <= 0:      # 分错才更新
                    self.w += self.lr * ti * xi
                    self.b += self.lr * ti
                    errors += 1
            if errors == 0:
                break
        return ep + 1

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)


# ---------- 2. MLP + 反向传播 ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)        # 数值稳定
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """全连接网络。sizes = [输入维, 隐层维..., 输出类数];隐层 sigmoid,输出 softmax。"""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        # Xavier 初始化:方差 2/(fan_in+fan_out),比全 0 好(打破对称性)
        self.W = [rng.normal(0, np.sqrt(2 / (i + o)), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, X):
        A = [X]                                   # A[0] = 输入
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A[-1] @ W + b
            A.append(softmax(Z) if l == len(self.W) - 1 else sigmoid(Z))
        return A

    def loss(self, A, Y):
        """交叉熵: -Σ y·log ŷ / n。输出已 softmax,直接取对应分量。"""
        n = len(Y)
        return float(-np.sum(Y * np.log(A[-1] + 1e-12)) / n)

    def backward(self, A, Y):
        """返回 (grads_W, grads_b),与 self.W/self.b 一一对应。"""
        n = len(Y)
        delta = (A[-1] - Y) / n                   # softmax+CE 输出层 delta
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = A[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:                             # 传到下一层(更靠输入)前过激活导数
                delta = (delta @ self.W[l].T) * A[l] * (1 - A[l])
        return gW, gb

    def train(self, X, Y, epochs=8000, lr=2.0, log_every=None):
        for ep in range(epochs):
            A = self.forward(X)
            gW, gb = self.backward(A, Y)
            for l in range(len(self.W)):          # 全批量梯度下降
                self.W[l] -= lr * gW[l]
                self.b[l] -= lr * gb[l]
            if log_every and (ep + 1) % log_every == 0:
                print(f"    epoch {ep+1:5d}  loss={self.loss(self.forward(X), Y):.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X)[-1], axis=1)


# ---------- 3. 梯度检查 ----------
def grad_check(model, X, Y, eps=1e-5):
    """数值梯度 vs 解析梯度,返回最大相对误差(应 < 1e-6 左右)。"""
    A = model.forward(X)
    gW, gb = model.backward(A, Y)

    def total_loss():
        return model.loss(model.forward(X), Y)

    worst = 0.0
    for params, grads in ((model.W, gW), (model.b, gb)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=["multi_index"])
            for _ in it:
                ix = it.multi_index
                old = P[ix]
                P[ix] = old + eps
                lp = total_loss()
                P[ix] = old - eps
                lm = total_loss()
                P[ix] = old
                num = (lp - lm) / (2 * eps)
                denom = max(abs(num) + abs(G[ix]), 1e-8)
                worst = max(worst, abs(num - G[ix]) / denom)
    return worst


if __name__ == "__main__":
    # XOR:线性不可分的教科书例子
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y01 = np.array([0, 1, 1, 0])
    Y = np.eye(2)[y01]                            # one-hot

    print("── [1] 感知机:AND 可分 ✔ / XOR 不可分 ✘ ──")
    p_and = Perceptron(2)
    n_ep = p_and.fit(X, np.array([0, 0, 0, 1]))
    print(f"  AND:{p_and.predict(X)}  收敛用了 {n_ep} 轮")
    p_xor = Perceptron(2)
    n_ep = p_xor.fit(X, y01, epochs=60)
    print(f"  XOR:{p_xor.predict(X)}  真值 [0 1 1 0],60 轮都不收敛(用了 {n_ep} 轮)")

    print("── [2] 梯度检查(先确认手推的反向传播是对的)──")
    net = MLP([2, 8, 2], seed=0)
    err = grad_check(net, X, Y)
    print(f"  最大相对误差 = {err:.2e}  ->  {'PASS ✔' if err < 1e-5 else 'FAIL ✘'}")
    assert err < 1e-5

    print("── [3] MLP 解 XOR(2 输入 -> 8 隐层 sigmoid -> 2 输出 softmax)──")
    net.train(X, Y, epochs=8000, lr=2.0, log_every=2000)
    print(f"  预测:{net.predict(X)}  真值:{y01}  准确率 {np.mean(net.predict(X) == y01):.0%}")
    print("\n结论:一层感知机卡死在 XOR,两层 MLP + 非线性轻松解决 —— 这就是『深度』的意义")
