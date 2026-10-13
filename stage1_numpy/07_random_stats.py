import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
案例 07：随机数与统计 — AI 实验的基础设施
=============================================

核心概念：
  - 随机数生成: AI 实验可复现的基础
  - 统计函数: 数据分析的核心工具
  - 分布: 正态分布、均匀分布在 AI 中的应用

Java 类比：
  Java:    Random rnd = new Random(seed)
  NumPy:   rng = np.random.default_rng(seed)

运行：python stage1_numpy/07_random_stats.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 随机数生成器")

    # 新版推荐写法: default_rng (NumPy >= 1.17)
    # 固定种子保证结果可复现 — AI 实验的基本要求
    rng = np.random.default_rng(seed=42)

    # 均匀分布 [0, 1)
    uniform = rng.random(5)
    print(f"rng.random(5) = {uniform}")

    # 指定范围和形状
    uniform_2d = rng.random((2, 3))
    print(f"rng.random((2,3)) =\n{uniform_2d}")

    # 正态分布（高斯分布）
    normal = rng.normal(loc=0, scale=1, size=5)  # 均值0, 标准差1
    print(f"rng.normal(0, 1, 5) = {normal}")

    # 整数随机数
    integers = rng.integers(0, 10, size=5)  # [0, 10) 范围
    print(f"rng.integers(0, 10, 5) = {integers}")

    # 可复现性验证
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    print(f"\n可复现性验证:")
    print(f"  rng1.random(3) = {rng1.random(3)}")
    print(f"  rng2.random(3) = {rng2.random(3)}")
    print(f"  相同种子 → 相同结果 OK")

    section("2. 常用分布")

    rng = np.random.default_rng(42)

    # 正态分布: 自然界最常见，AI 中权重初始化首选
    normal_data = rng.normal(loc=0, scale=1, size=10000)
    print(
        f"正态分布 N(0,1): mean={normal_data.mean():.4f}, std={normal_data.std():.4f}"
    )
    print(f"  范围: [{normal_data.min():.2f}, {normal_data.max():.2f}]")

    # 均匀分布
    uniform_data = rng.uniform(-1, 1, size=10000)
    print(
        f"均匀分布 U(-1,1): mean={uniform_data.mean():.4f}, std={uniform_data.std():.4f}"
    )

    # 二项分布（抛硬币）
    coins = rng.binomial(n=1, p=0.5, size=10000)  # 10000次抛硬币
    print(f"二项分布 B(1,0.5): 正面比例={coins.mean():.4f}")

    # choice: 从数组中随机选择
    items = np.array(["A", "B", "C", "D"])
    sampled = rng.choice(items, size=5, replace=True)  # 有放回抽样
    print(f"随机抽样: {sampled}")

    # 不放回抽样（类似 Java Collections.shuffle 后取前N个）
    sampled_noreplace = rng.choice(items, size=3, replace=False)
    print(f"不放回抽样: {sampled_noreplace}")

    section("3. 描述性统计")

    data = np.array([23, 19, 31, 27, 22, 29, 25, 30, 24, 28])

    print(f"数据: {data}")
    print(f"  均值 mean    = {data.mean():.2f}")
    print(f"  中位数 median = {np.median(data):.2f}")
    print(f"  标准差 std    = {data.std():.2f}")
    print(f"  方差 var      = {data.var():.2f}")
    print(f"  最小值 min    = {data.min()}")
    print(f"  最大值 max    = {data.max()}")
    print(f"  极差 range    = {data.max() - data.min()}")
    print(f"  分位数 25%    = {np.percentile(data, 25):.2f}")
    print(f"  分位数 75%    = {np.percentile(data, 75):.2f}")

    # Java 等价:
    # mean: Arrays.stream(data).average().getAsDouble()
    # max: Arrays.stream(data).max().getAsInt()

    section("4. 沿轴统计 — 二维数据")

    matrix = np.array(
        [
            [85, 92, 78],
            [70, 88, 95],
            [90, 76, 82],
            [65, 70, 72],
            [88, 95, 91],
        ]
    )
    print(f"成绩矩阵 (5学生x3科目):\n{matrix}")

    print(f"\n--- 每列统计（按科目）---")
    print(f"  每科均值: {matrix.mean(axis=0)}")
    print(f"  每科最高: {matrix.max(axis=0)}")
    print(f"  每科最低: {matrix.min(axis=0)}")
    print(f"  每科标准差: {np.array2string(matrix.std(axis=0), precision=2)}")

    print(f"\n--- 每行统计（按学生）---")
    print(f"  每人均值: {matrix.mean(axis=1)}")
    print(f"  每人最高: {matrix.max(axis=1)}")

    print("""
  axis 口诀:
  axis=0 → 沿"行"方向操作 → 消灭行维度 → 结果是每列的统计
  axis=1 → 沿"列"方向操作 → 消灭列维度 → 结果是每行的统计
  形象记忆: axis=0 的结果少了一行""")

    section("5. 累积运算")

    arr = np.array([1, 2, 3, 4, 5])
    print(f"arr = {arr}")

    # 累积和
    print(f"cumsum  = {np.cumsum(arr)}")  # [1 3 6 10 15]
    print(f"cumprod = {np.cumprod(arr)}")  # [1 2 6 24 120]

    # 差分（后项减前项）
    diff = np.diff(arr)
    print(f"diff    = {diff}")  # [1 1 1 1]

    section("6. 排序与去重")

    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    print(f"arr = {arr}")

    # 排序（返回新数组，不修改原数组）
    sorted_arr = np.sort(arr)
    print(f"np.sort = {sorted_arr}")

    # 排序索引（返回排序后的索引顺序）
    sort_idx = np.argsort(arr)
    print(f"np.argsort = {sort_idx}")
    print(f"用索引排序: {arr[sort_idx]}")

    # 去重并排序
    unique = np.unique(arr)
    print(f"np.unique = {unique}")

    # 去重 + 计数
    unique, counts = np.unique(arr, return_counts=True)
    print(f"去重: {unique}")
    print(f"计数: {counts}")

    section("7. 实战：模拟权重初始化")

    # 神经网络权重初始化是 AI 中随机数最核心的应用
    # 好的初始化能让训练更快收敛

    rng = np.random.default_rng(42)

    # Xavier/Glorot 初始化: std = sqrt(2 / (fan_in + fan_out))
    fan_in, fan_out = 3, 4
    xavier_std = np.sqrt(2.0 / (fan_in + fan_out))
    W_xavier = rng.normal(0, xavier_std, (fan_in, fan_out))
    print(f"Xavier 初始化 (fan_in={fan_in}, fan_out={fan_out}):")
    print(f"  std = {xavier_std:.4f}")
    print(f"  W =\n{W_xavier}")
    print(f"  实际 std = {W_xavier.std():.4f}")

    # He/Kaiming 初始化（ReLU 激活函数用）
    he_std = np.sqrt(2.0 / fan_in)
    W_he = rng.normal(0, he_std, (fan_in, fan_out))
    print(f"\nHe 初始化 (fan_in={fan_in}):")
    print(f"  std = {he_std:.4f}")
    print(f"  W =\n{W_he}")

    print("""
  *** 为什么权重初始化很重要 ***
  - 太小 → 前向传播信号衰减，梯度消失
  - 太大 → 前向传播信号爆炸，梯度爆炸
  - Xavier/He 初始化根据输入/输出维度自适应选择方差
  - 这也是为什么理解"分布"和"标准差"对 AI 很重要""")

    print("\n[OK] 案例 07 完成 - 随机数和统计是 AI 实验的基础设施")
    print("     关键认知: 固定种子保证可复现; 权重初始化是随机数的核心应用")


if __name__ == "__main__":
    main()
