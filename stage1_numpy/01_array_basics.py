import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
案例 01：ndarray 基础 — Python AI 的地基
=============================================

核心概念：
  - ndarray 是 NumPy 的核心数据结构，类比 Java 的 double[]，但支持多维
  - 所有元素同一类型（dtype），内存连续，这是高性能的基础
  - shape 描述每个维度的长度

Java 类比：
  Java            double[] arr = new double[4]
  Python          arr = np.array([1.0, 2.0, 3.0, 4.0])

  Java 二维       double[][] matrix = new double[3][4]
  NumPy 2D        matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

运行：python stage1_numpy/01_array_basics.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. 创建一维数组")

    # 从 Python list 创建
    arr1 = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"arr1 = {arr1}")
    print(f"  shape = {arr1.shape}")  # (4,)
    print(f"  dtype = {arr1.dtype}")  # float64
    print(f"  ndim  = {arr1.ndim}")  # 1

    # Java 开发者注意：Python list 是动态类型、可混合类型
    # 但 ndarray 创建后类型固定，类似 Java 原始数组

    section("2. 创建多维数组")

    # 2D: 类比 Java 的 double[][]
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"matrix =\n{matrix}")
    print(f"  shape = {matrix.shape}")  # (3, 4) — 3行4列
    print(f"  ndim  = {matrix.ndim}")  # 2

    # 3D: 类比 Java 的 double[][][]
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"\ntensor shape = {tensor.shape}")  # (2, 2, 2)
    print(f"tensor ndim  = {tensor.ndim}")  # 3

    section("3. 工厂函数创建数组")

    # Java 里你写 new double[5] 得到全 0 数组
    # NumPy 提供更多工厂函数

    zeros = np.zeros(5)
    print(f"np.zeros(5)       = {zeros}")

    zeros_2d = np.zeros((3, 4))
    print(f"np.zeros((3,4)) shape = {zeros_2d.shape}")

    ones = np.ones((2, 3))
    print(f"np.ones((2,3)) =\n{ones}")

    # arange 类比 Java 的 for(int i=0; i<10; i++)
    seq = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
    print(f"np.arange(0,10,2) = {seq}")

    # linspace: 均匀分布的数字，常用于生成采样点
    lin = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
    print(f"np.linspace(0,1,5) = {lin}")

    # 随机数（AI 里无处不在）
    rng = np.random.default_rng(42)  # 固定种子保证可复现
    rand_arr = rng.random((2, 3))
    print(f"rng.random((2,3)) =\n{rand_arr}")

    section("4. dtype — 数据类型")

    # ndarray 是强类型的，类似 Java int[] vs double[]
    int_arr = np.array([1, 2, 3])
    float_arr = np.array([1.0, 2.0, 3.0])
    print(f"int_arr.dtype   = {int_arr.dtype}")  # int64
    print(f"float_arr.dtype = {float_arr.dtype}")  # float64

    # 显式指定类型
    f32 = np.array([1, 2, 3], dtype=np.float32)
    print(f"dtype=float32   = {f32.dtype}")

    # 类型转换：astype，类比 Java 的 (double) cast
    converted = int_arr.astype(np.float64)
    print(f"astype(float64) = {converted.dtype}")

    section("5. shape 和 ndim 的理解")

    # shape 是一个元组，每个数字代表一个维度的大小
    # ndim = len(shape)
    shapes = [
        np.array([1, 2, 3]),  # (3,)     1D
        np.array([[1, 2], [3, 4]]),  # (2, 2)   2D
        np.array([[[1], [2]], [[3], [4]]]),  # (2, 2, 1) 3D
    ]
    for i, s in enumerate(shapes):
        print(f"  示例{i + 1}: shape={s.shape}, ndim={s.ndim}")

    print("\n[OK] 案例 01 完成 - 你现在理解了 ndarray 的基本结构")
    print("     关键认知: shape/dtype/ndim 是 ndarray 的三个核心属性")


if __name__ == "__main__":
    main()
