import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
案例 04：布尔索引与花式索引 — 数据筛选利器
=============================================

核心概念：
  - 布尔索引：用条件表达式作为数组索引，选出满足条件的元素
  - 花式索引：用整数数组作为索引，按指定顺序取出元素
  - 这是 Java Stream filter / map 的 NumPy 等价物

Java 类比：
  Java: list.stream().filter(x -> x > 5).collect(...)
  NumPy: arr[arr > 5]

运行：python stage1_numpy/04_boolean_indexing.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 布尔索引基础")

    arr = np.array([3, 7, 1, 8, 5, 9, 2, 6])
    print(f"arr = {arr}")

    # 条件表达式返回布尔数组
    mask = arr > 5
    print(f"arr > 5 = {mask}")
    # [False True False True False True False True]

    # 用布尔数组做索引，只返回 True 位置的元素
    print(f"arr[arr > 5] = {arr[arr > 5]}")  # [7 8 9 6]

    print("""
  Java 等价写法:
    List<Integer> result = new ArrayList<>();
    for (int x : arr) if (x > 5) result.add(x);

  NumPy 用一行 + 底层 C 运算完成""")

    section("2. 组合条件")

    # & (与), | (或), ~ (非) — 注意必须用括号！
    print(f"arr > 3:  {arr > 3}")
    print(f"arr < 7:  {arr < 7}")

    # 3 < arr < 7 → (arr > 3) & (arr < 7)
    result = arr[(arr > 3) & (arr < 7)]
    print(f"(arr > 3) & (arr < 7) = {result}")  # [5 6]

    # 或条件
    result_or = arr[(arr < 3) | (arr > 8)]
    print(f"(arr < 3) | (arr > 8) = {result_or}")  # [1 9 2]

    # 非
    result_not = arr[~(arr > 5)]
    print(f"~(arr > 5) = {result_not}")  # [3 1 5 2]

    section("3. 布尔索引赋值 — 原地修改")

    data = np.array([10, 20, 30, 40, 50])
    print(f"原数组: {data}")

    # 把所有大于 25 的元素替换为 0
    data[data > 25] = 0
    print(f"把>25的设为0: {data}")  # [10 20 0 0 0]

    # Java 等价:
    # for (int i = 0; i < data.length; i++)
    #     if (data[i] > 25) data[i] = 0;

    section("4. 二维数组的布尔索引")

    matrix = np.array(
        [
            [85, 92, 78],
            [70, 88, 95],
            [90, 76, 82],
            [65, 70, 72],
            [88, 95, 91],
        ]
    )
    print(f"成绩矩阵 (5学生 x 3科目):\n{matrix}")

    # 选出所有 90 分以上的
    high = matrix[matrix >= 90]
    print(f"\n>=90 的成绩: {high}")

    # 按行筛选：找出每科都 >= 80 的学生
    all_good = (matrix >= 80).all(axis=1)  # 沿列方向检查
    print(f"\n每科>=80的学生: {all_good}")  # [False False True False True]
    print(f"这些学生的成绩:\n{matrix[all_good]}")

    # 按列筛选：找出平均分 > 80 的科目
    col_avg = matrix.mean(axis=0)
    good_subjects = col_avg > 80
    print(f"\n各科目平均分: {col_avg}")
    print(f"平均>80的科目: {good_subjects}")

    section("5. where — 条件查找")

    arr = np.array([3, 7, 1, 8, 5, 9, 2, 6])

    # np.where(condition) 返回满足条件的索引
    indices = np.where(arr > 5)
    print(f"arr = {arr}")
    print(f"np.where(arr > 5) = {indices}")
    print(f"满足条件的元素: {arr[indices]}")

    # np.where(cond, x, y) — 三元表达式，类比 Java 三元运算符
    # Java: x > 5 ? "高" : "低"
    labels = np.where(arr > 5, "高", "低")
    print(f"np.where(arr>5, '高', '低') = {labels}")

    section("6. 花式索引 — 按指定顺序取元素")

    arr = np.array([10, 20, 30, 40, 50])
    print(f"arr = {arr}")

    # 用整数数组做索引
    idx = np.array([4, 2, 0, 1])
    print(f"arr[[4,2,0,1]] = {arr[idx]}")  # [50 30 10 20]

    # 二维花式索引
    matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    print(f"\nmatrix =\n{matrix}")

    # 取指定行
    rows = np.array([0, 2])
    print(f"matrix[[0,2]] =\n{matrix[rows]}")

    # 取指定行列交叉点
    cols = np.array([0, 2])
    print(f"matrix[[0,2],[0,2]] = {matrix[rows, cols]}")  # [1 9]

    # 对比 Java:
    # List<Integer> result = Arrays.asList(matrix[0][0], matrix[2][2])

    section("7. 实战：数据清洗与筛选")

    # 模拟 10 条传感器数据，其中有些异常值
    readings = np.array([22.5, 23.1, 999.0, 22.8, 23.0, -50.0, 22.9, 23.2, 998.0, 23.1])
    print(f"原始读数: {readings}")

    # 检测异常值（假设正常范围 20-25）
    is_normal = (readings >= 20) & (readings <= 25)
    print(f"正常数据: {readings[is_normal]}")
    print(f"异常数据: {readings[~is_normal]}")

    # 用中位数替换异常值
    median = np.median(readings[is_normal])
    cleaned = readings.copy()
    cleaned[~is_normal] = median
    print(f"清洗后 (异常替换为中位数 {median}): {cleaned}")

    print("\n[OK] 案例 04 完成 - 布尔索引是数据筛选和清洗的核心工具")
    print("     关键认知: 条件表达式生成 mask，mask 当索引用")


if __name__ == "__main__":
    main()
