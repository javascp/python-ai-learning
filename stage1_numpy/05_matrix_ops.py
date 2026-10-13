import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
案例 05：矩阵运算 — 神经网络的核心数学
=============================================

核心概念：
  - 矩阵乘法 (@) 是神经网络前向传播和反向传播的基础
  - 矩阵转置 (T) 在梯度计算中无处不在
  - 理解 shape 变化是掌握深度学习的关键

Java 类比：
  Java 三层 for 循环手写矩阵乘法:
    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
            for (int k=0; k<K; k++)
                C[i][j] += A[i][k] * B[k][j];
  NumPy: C = A @ B

运行：python stage1_numpy/05_matrix_ops.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 矩阵乘法基础")

    A = np.array([[1, 2], [3, 4]])  # 2x2
    B = np.array([[5, 6], [7, 8]])  # 2x2

    # @ 是矩阵乘法运算符（Python 3.5+）
    # 也可用 np.dot(A, B) 或 A.dot(B)
    C = A @ B
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A @ B =\n{C}")

    print("""
  手动验证 C[0][0] = 1*5 + 2*7 = 21 OK
  手动验证 C[0][1] = 1*6 + 2*8 = 22 OK""")

    section("2. 矩阵乘法的 shape 规则")

    # A shape (M, K) @ B shape (K, N) = C shape (M, N)
    # 关键：A 的列数必须等于 B 的行数

    shapes = [
        ((2, 3), (3, 4), "2x3 @ 3x4 = 2x4"),
        ((4, 5), (5, 3), "4x5 @ 5x3 = 4x3"),
        ((1, 3), (3, 1), "1x3 @ 3x1 = 1x1 (点积)"),
        ((3, 1), (1, 3), "3x1 @ 1x3 = 3x3 (外积)"),
    ]
    for sa, sb, desc in shapes:
        A = np.random.rand(*sa)
        B = np.random.rand(*sb)
        C = A @ B
        print(f"  {desc}: {A.shape} @ {B.shape} = {C.shape}")

    section("3. 逐元素乘 vs 矩阵乘法 — 别混淆！")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # * 是逐元素乘 (Hadamard积)
    elementwise = A * B
    print(f"A * B (逐元素) =\n{elementwise}")
    # [[5 12]
    #  [21 32]]

    # @ 是矩阵乘法
    matmul = A @ B
    print(f"A @ B (矩阵乘) =\n{matmul}")
    # [[19 22]
    #  [43 50]]

    print("""
  *** 重要区分 ***
  A * B → 对应位置相乘，要求 shape 完全相同
  A @ B → 矩阵乘法，要求 A列数 = B行数
  深度学习中两种都会用到，搞混会导致 bug""")

    section("4. 转置 — 梯度计算的基础")

    A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
    print(f"A =\n{A}")
    print(f"A.shape = {A.shape}")

    # 转置：行列互换
    AT = A.T
    print(f"A.T =\n{AT}")
    print(f"A.T.shape = {AT.shape}")  # 3x2

    # 矩阵乘法结合转置
    # 在神经网络反向传播中: dW = dY.T @ X 这样的形式很常见
    X = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    dY = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3x2
    dW = dY.T @ X  # (2x3) @ (3x2) = (2x2)
    print(f"\nX =\n{X}")
    print(f"dY =\n{dY}")
    print(f"dY.T @ X =\n{dW}")
    print(f"dW.shape = {dW.shape}")

    section("5. 实战：模拟神经网络前向传播")

    # 模拟一个 2 层神经网络
    # 输入层: 3 个特征
    # 隐藏层: 4 个神经元
    # 输出层: 2 个输出

    rng = np.random.default_rng(42)

    # --- 单样本前向传播 ---
    x = np.array([1.0, 2.0, 3.0])  # 输入, shape (3,)

    # 第一层权重和偏置
    W1 = rng.normal(0, 0.5, (3, 4))  # 3输入 -> 4隐藏, shape (3, 4)
    b1 = np.array([0.1, 0.1, 0.1, 0.1])  # 4个偏置

    # 第二层权重和偏置
    W2 = rng.normal(0, 0.5, (4, 2))  # 4隐藏 -> 2输出, shape (4, 2)
    b2 = np.array([0.2, 0.2])  # 2个偏置

    # 前向传播
    # 第一层: z1 = x @ W1 + b1, shape (3,) @ (3,4) = (4,) + (4,) = (4,)
    z1 = x @ W1 + b1
    # 激活函数: ReLU (后面会详细讲)
    a1 = np.maximum(0, z1)  # ReLU: max(0, x)

    # 第二层: z2 = a1 @ W2 + b2
    z2 = a1 @ W2 + b2
    # 输出
    print(f"输入 x: {x}")
    print(f"  shape = {x.shape}")
    print(f"第一层 z1 = x @ W1 + b1: {z1}")
    print(f"  shape = {z1.shape}")
    print(f"ReLU(a1) = {a1}")
    print(f"第二层 z2 = a1 @ W2 + b2: {z2}")
    print(f"  shape = {z2.shape}")
    print(f"最终输出: {z2}")

    # --- 批量前向传播（5 个样本同时算）---
    print("\n--- 批量前向传播 ---")
    X_batch = rng.random((5, 3))  # 5 个样本，每个 3 特征

    # 批量计算: X_batch @ W1 + b1
    # (5,3) @ (3,4) = (5,4), b1 (4,) 广播到 (5,4)
    Z1 = X_batch @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2

    print(f"X_batch shape: {X_batch.shape} (5样本, 3特征)")
    print(f"Z1 shape: {Z1.shape} (5样本, 4隐藏)")
    print(f"Z2 shape: {Z2.shape} (5样本, 2输出)")
    print(f"输出:\n{Z2}")

    print("""
  *** 关键洞察 ***
  单样本和批量的唯一区别: x (3,) → X (5,3)
  矩阵乘法自动处理了批量维度，这就是为什么
  深度学习用矩阵而非循环""")

    section("6. 常用线性代数函数")

    A = np.array([[4, 2], [1, 3]])
    print(f"A =\n{A}")
    print(f"  对角线元素 np.diag(A) = {np.diag(A)}")
    print(f"  迹 np.trace(A) = {np.trace(A)}")  # 4+3=7
    print(f"  行列式 np.linalg.det(A) = {np.linalg.det(A):.1f}")  # 4*3-2*1=10
    print(f"  逆矩阵 np.linalg.inv(A) =\n{np.linalg.inv(A)}")
    print(f"  A @ inv(A) =\n{A @ np.linalg.inv(A).round(10)}")  # 应该接近单位阵

    # 解线性方程组 Ax = b
    b = np.array([10, 8])
    x = np.linalg.solve(A, b)
    print(f"\n解方程组 Ax=b: x = {x}")  # [1 3] → 4*1+2*3=10, 1*1+3*3=10
    print(f"  验证 A @ x = {A @ x}")  # 应该 = [10 8]

    print("\n[OK] 案例 05 完成 - 矩阵运算是神经网络的核心")
    print("     关键认知: y = W @ x + b 这一行代码就是一层神经网络")
    print("     批量处理 = 在最前面加一个维度，矩阵乘法自动处理")


if __name__ == "__main__":
    main()
