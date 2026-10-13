"""
NumPy 练习题 —— 第一阶段第 1 课
用法: python exercise_numpy.py
每题有 TODO 标记，取消注释并补全代码后运行验证
"""

import numpy as np


# ============================================================
# 练习 1：创建数组与基本运算
# ============================================================
# 任务: 创建一个 3x3 的矩阵，内容为 1 到 9，然后:
#   (a) 每个元素乘以 10
#   (b) 求所有元素的和、每列的和、每行的和


def exercise_1():
    print("=" * 50)
    print("练习 1：创建数组与基本运算")
    print("=" * 50)

    # TODO: 用 np.array 创建 3x3 矩阵 [[1,2,3],[4,5,6],[7,8,9]]
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # TODO: 每个元素乘以 10（向量化，不要用循环）
    multiplied = matrix * 10
    print(f"乘以 10:\n{multiplied}")

    # TODO: 所有元素的和
    total = matrix.sum()
    print(f"总和: {total}")  # 应该是 45

    # TODO: 每列的和（axis=?）
    col_sums = matrix.sum(axis=0)
    print(f"每列和: {col_sums}")  # 应该是 [12 15 18]

    # TODO: 每行的和
    row_sums = matrix.sum(axis=1)
    print(f"每行和: {row_sums}")  # 应该是 [6 15 24]


# ============================================================
# 练习 2：广播与布尔索引
# ============================================================
# 任务: 给定学生成绩矩阵（5个学生，3门课），完成统计分析


def exercise_2():
    print("\n" + "=" * 50)
    print("练习 2：广播与布尔索引")
    print("=" * 50)

    # 5个学生 × 3门课的成绩
    scores = np.array(
        [
            [85, 92, 78],  # 学生0
            [70, 88, 95],  # 学生1
            [90, 76, 82],  # 学生2
            [65, 70, 72],  # 学生3
            [88, 95, 91],  # 学生4
        ]
    )

    # TODO (a): 用广播给每门课加 5 分（一个向量加到矩阵上）
    bonus = np.array([5, 5, 5])
    adjusted = scores + bonus
    print(f"加5分后:\n{adjusted}")

    # TODO (b): 求每个学生的平均分（axis=?）
    student_avg = adjusted.mean(axis=1)
    print(f"每个学生平均分: {student_avg}")

    # TODO (c): 求每门课的最高分
    subject_max = adjusted.max(axis=0)
    print(f"每门课最高分: {subject_max}")

    # TODO (d): 用布尔索引找出所有 90 分以上的成绩
    high_scores = adjusted[adjusted >= 90]
    print(f"90分以上的成绩: {high_scores}")

    # TODO (e): 找出平均分超过 85 的学生编号
    # 提示: 先算平均分，再用布尔条件做索引
    good_students = np.where(adjusted.mean(axis=1) > 85)[0]
    print(f"平均分>85的学生编号: {good_students}")


# ============================================================
# 练习 3：矩阵运算（神经网络的基础）
# ============================================================
# 任务: 模拟神经网络一层的计算: y = W @ x + b
#   这是所有深度学习框架最基本的操作


def exercise_3():
    print("\n" + "=" * 50)
    print("练习 3：矩阵运算（模拟神经网络一层）")
    print("=" * 50)

    # 输入向量 x: 3 个特征
    x = np.array([1.0, 2.0, 3.0])

    # 权重矩阵 W: 4 个神经元，每个接收 3 个输入
    W = np.array(
        [
            [0.1, 0.2, -0.3],
            [-0.5, 0.4, 0.2],
            [0.3, -0.1, 0.5],
            [0.2, 0.3, -0.4],
        ]
    )

    # 偏置向量 b: 每个神经元一个偏置
    b = np.array([0.1, -0.2, 0.05, 0.3])

    # TODO: 计算 y = W @ x + b（矩阵乘向量 + 广播加偏置）
    # 提示: W @ x 得到 4 个值，再加 b
    y = W @ x + b
    print(f"输入 x: {x}")
    print(f"权重 W shape: {W.shape}")  # (4, 3)
    print(f"输出 y: {y}")  # 4 个值
    print(f"y shape: {y.shape}")  # (4,)

    # 验证: 手动算第一个神经元
    # y[0] = 0.1*1 + 0.2*2 + (-0.3)*3 + 0.1 = 0.1 + 0.4 - 0.9 + 0.1 = -0.3
    print(f"验证 y[0] = {0.1 * 1 + 0.2 * 2 + (-0.3) * 3 + 0.1:.4f} (应为 -0.3)")


# ============================================================
# 练习 4：性能对比（体感向量化优势）
# ============================================================
def exercise_4():
    print("\n" + "=" * 50)
    print("练习 4：性能对比")
    print("=" * 50)

    size = 5_000_000
    py_list = list(range(size))
    np_arr = np.arange(size)

    # Python 列表循环
    import time

    start = time.time()
    py_result = [x * 2 + 1 for x in py_list]
    py_time = time.time() - start
    print(f"Python list 循环 ({size:,} 个元素): {py_time:.3f}s")

    # NumPy 向量化
    start = time.time()
    np_result = np_arr * 2 + 1
    np_time = time.time() - start
    print(f"NumPy 向量化      ({size:,} 个元素): {np_time:.3f}s")

    if py_time > 0:
        print(f"加速比: {py_time / np_time:.1f}x")


# ============================================================
# 练习 5：reshape 与切片（数据形状变换）
# ============================================================
def exercise_5():
    print("\n" + "=" * 50)
    print("练习 5：reshape 与切片")
    print("=" * 50)

    # 一维数组 0-23
    data = np.arange(24)
    print(f"原始: {data}")

    # TODO (a): reshape 成 4x6 矩阵
    matrix = data.reshape(4, 6)
    print(f"4x6:\n{matrix}")

    # TODO (b): reshape 成 2x3x4 三维数组（2个批次，每批3行4列）
    tensor = data.reshape(2, 3, 4)
    print(f"2x3x4:\n{tensor}")

    # TODO (c): 取出第 2 行第 3-5 列的元素
    print(f"matrix[1, 2:5] = {matrix[1, 2:5]}")

    # TODO (d): 取出最后两行
    print(f"matrix[-2:] =\n{matrix[-2:]}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print(
        "\n[OK] 全部练习完成！如果结果和注释里的预期值一致，说明你已掌握 NumPy 基础。"
    )
