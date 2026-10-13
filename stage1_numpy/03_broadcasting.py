import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
案例 03：广播 (Broadcasting) — 形状不匹配时的自动扩展
=============================================

核心概念：
  - 广播让不同形状的数组进行运算时，自动"扩展"小数组以匹配大数组
  - 不实际复制数据，只是逻辑上的扩展
  - 这是 NumPy 最强大也最容易让 Java 开发者困惑的特性

Java 类比：
  Java 里如果你想给矩阵每行加一个向量，需要：
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] += vector[j];
  NumPy 只需：matrix + vector

运行：python stage1_numpy/03_broadcasting.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 标量广播")

    a = np.array([[1, 2, 3], [4, 5, 6]])
    # 标量 10 "广播"到 2x3 的每个元素
    print(f"a =\n{a}")
    print(f"a + 10 =\n{a + 10}")
    print("  --> 标量自动扩展到所有元素")

    section("2. 一维数组广播到二维")

    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    vector = np.array([10, 20, 30, 40])  # 长度 4 = matrix 的列数

    # vector 被自动扩展为 3x4，然后逐元素相加
    result = matrix + vector
    print(f"matrix (3x4) =\n{matrix}")
    print(f"vector (4,)  = {vector}")
    print(f"matrix + vector =\n{result}")
    print("  --> vector 从 (4,) 广播到 (3,4)，每行加同样的向量")

    section("3. 列向量广播")

    # reshape(-1, 1) 把一维数组变成列向量
    col_vector = np.array([100, 200, 300]).reshape(-1, 1)  # shape (3, 1)
    print(f"col_vector (3,1) =\n{col_vector}")

    # 列向量广播到矩阵的每一列
    result2 = matrix + col_vector
    print(f"matrix + col_vector =\n{result2}")
    print("  --> col_vector 从 (3,1) 广播到 (3,4)，每列加同样的值")

    section("4. 广播规则 — 三步判断法")

    print("""
  广播规则（从右往左对齐维度）：
  1. 两个数组 shape 从右往左逐维比较
  2. 每维相等，或其中一个为 1，则兼容
  3. 为 1 的维度被"扩展"到另一个数组的对应维度

  示例:
  A.shape = (4, 3)    B.shape = (3,)     → B 广播为 (4, 3) OK
  A.shape = (4, 3)    B.shape = (4,)     → 不兼容！因为 B 最后维度 4 ≠ 3
  A.shape = (4, 3)    B.shape = (4, 1)   → B 广播为 (4, 3) OK
  A.shape = (3,)      B.shape = (4, 3)   → A 广播为 (4, 3) OK
  """)

    # 验证不兼容的情况
    try:
        bad = np.array([1, 2, 3, 4])  # shape (4,)
        good_matrix = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
        result = good_matrix + bad  # 4 ≠ 3，应该报错
    except ValueError as e:
        print(f"  不兼容示例: {e}")

    section("5. 实战：标准化数据矩阵")

    # 模拟 5 个样本，3 个特征的数据
    # 类比: 5 条用户数据，每条有年龄、收入、消费 3 个特征
    data = np.array(
        [
            [25, 5000, 800],
            [30, 8000, 1200],
            [35, 12000, 3000],
            [40, 15000, 3500],
            [45, 20000, 5000],
        ]
    )

    print(f"原始数据 (5样本 x 3特征):\n{data}")
    print(f"  每列均值 = {data.mean(axis=0)}")
    print(f"  每列标准差 = {np.array2string(data.std(axis=0), precision=2)}")

    # Z-score 标准化：每列减去该列均值，除以该列标准差
    # 均值向量的 shape 是 (3,)，会广播到 (5, 3)
    means = data.mean(axis=0)  # shape (3,)
    stds = data.std(axis=0)  # shape (3,)
    standardized = (data - means) / stds

    print(f"\n标准化后:\n{standardized}")
    print(f"  标准化后每列均值: {standardized.mean(axis=0).round(10)}")
    print(f"  标准化后每列标准差: {standardized.std(axis=0).round(10)}")
    print("\n  --> 这就是广播在机器学习数据预处理中的实际用途")

    section("6. 广播在神经网络中的应用预览")

    # 神经网络一层的计算: y = W @ x + b
    # W shape: (4, 3)  — 4个神经元，每个3个输入
    # x shape: (3,)    — 3个输入特征
    # W @ x shape: (4,) — 4个输出
    # b shape: (4,)    — 4个偏置
    # 加法 (4,) + (4,) 是逐元素，不需要广播

    # 但批量处理时:
    # X shape: (5, 3)  — 5个样本，每个3特征
    # W shape: (3, 4)  — 3输入，4个神经元
    # X @ W shape: (5, 4) — 5个样本各4个输出
    # b shape: (4,)   — 4个偏置
    # (5, 4) + (4,) → b 广播到 (5, 4) OK

    X = np.random.default_rng(42).random((5, 3))
    W = np.random.default_rng(43).random((3, 4))
    b = np.array([0.1, 0.2, 0.3, 0.4])

    Y = X @ W + b  # 广播让 b 自动加到每个样本上
    print(f"X shape: {X.shape} (5个样本, 3特征)")
    print(f"W shape: {W.shape} (3输入, 4神经元)")
    print(f"b shape: {b.shape} (4个偏置)")
    print(f"Y = X @ W + b shape: {Y.shape} (5个样本, 4输出)")
    print("  --> b 从 (4,) 广播到 (5,4)，每个样本都加了同样的偏置")

    print("\n[OK] 案例 03 完成 - 广播是神经网络批量计算的基础")
    print("     关键认知: 广播让不同形状的数组自动对齐，省去显式循环")


if __name__ == "__main__":
    main()
