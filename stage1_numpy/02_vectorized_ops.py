import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
案例 02：向量化运算 — NumPy 的性能密码
=============================================

核心概念：
  - 向量化 = 用一个操作作用于整个数组，替代 Python for 循环
  - 底层是 C 语言实现的 SIMD 批量指令，接近硬件极限
  - 这是 Python 能做 AI 的根本原因：Python 慢，但 NumPy 快

Java 类比：
  Java 循环:     for (int i=0; i<arr.length; i++) result[i] = arr[i] * 2;
  NumPy 向量化:  result = arr * 2

运行：python stage1_numpy/02_vectorized_ops.py
"""

import numpy as np
import time


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 逐元素运算")

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])

    # 以下每种运算，Java 里都要写 for 循环
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b  = {a + b}")  # 加
    print(f"a * b  = {a * b}")  # 逐元素乘（不是矩阵乘！）
    print(f"a - b  = {a - b}")  # 减
    print(f"a / b  = {a / b}")  # 除
    print(f"a ** 2 = {a**2}")  # 幂运算
    print(f"np.sqrt(a) = {np.sqrt(a)}")  # 开方

    # 标量运算（广播的一种）
    print(f"\na * 2   = {a * 2}")
    print(f"a + 100 = {a + 100}")

    section("2. 聚合运算")

    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"arr = {arr}")
    print(f"  sum   = {arr.sum()}")  # 求和
    print(f"  mean  = {arr.mean():.2f}")  # 均值
    print(f"  max   = {arr.max()}")  # 最大值
    print(f"  min   = {arr.min()}")  # 最小值
    print(f"  std   = {arr.std():.2f}")  # 标准差
    print(f"  argmax = {arr.argmax()}")  # 最大值的索引
    print(f"  argmin = {arr.argmin()}")  # 最小值的索引

    # Java 类比: Stream API
    # Java: Arrays.stream(arr).sum()
    # Java: Arrays.stream(arr).average().getAsDouble()

    section("3. 通用函数 (ufunc)")

    # ufunc = universal function，对整个数组逐元素操作的函数
    x = np.array([0, np.pi / 2, np.pi])
    print(f"x = {x}")
    print(f"np.sin(x)  = {np.sin(x)}")  # [0, 1, 0]
    print(f"np.cos(x)  = {np.cos(x)}")
    print(f"np.exp(x)  = {np.exp(np.array([0, 1, 2]))}")  # e^x
    print(f"np.log(x)  = {np.log(np.array([1, np.e, np.e**2]))}")

    section("4. 性能对比 — 体感差距")

    sizes = [100_000, 1_000_000, 5_000_000]
    print(f"{'数据量':>12} | {'Python循环':>12} | {'NumPy向量化':>12} | {'加速比':>8}")
    print("-" * 55)

    for size in sizes:
        py_data = list(range(size))
        np_data = np.arange(size)

        # Python 原生循环
        start = time.time()
        py_result = [x * 2.0 + 1.0 for x in py_data]
        py_time = time.time() - start

        # NumPy 向量化
        start = time.time()
        np_result = np_data * 2.0 + 1.0
        np_time = time.time() - start

        speedup = py_time / np_time if np_time > 0 else float("inf")
        print(f"{size:>12,} | {py_time:>10.4f}s | {np_time:>10.4f}s | {speedup:>7.1f}x")

    print("\n  --> 数据量越大，向量化优势越明显")
    print("  --> AI 训练处理百万/十亿级数据，向量化是必须的")

    section("5. 链式向量化运算")

    # 多个运算可以链式组合，全程不需要循环
    data = np.random.default_rng(42).random(1_000_000)

    # 标准化 (z-score normalization): (x - mean) / std
    # 这是机器学习最常见的数据预处理
    normalized = (data - data.mean()) / data.std()
    print(f"原始数据  mean={data.mean():.4f}, std={data.std():.4f}")
    print(f"标准化后 mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    print("  --> 标准化后均值约为 0，标准差约为 1")

    # Min-Max 归一化: (x - min) / (max - min)
    minmax = (data - data.min()) / (data.max() - data.min())
    print(f"归一化后 min={minmax.min():.4f}, max={minmax.max():.4f}")
    print("  --> 归一化后所有值缩放到 [0, 1] 区间")

    print("\n[OK] 案例 02 完成 - 向量化是 NumPy 的灵魂")
    print("     关键认知: 永远优先用向量化，而不是 for 循环遍历 ndarray")


if __name__ == "__main__":
    main()
