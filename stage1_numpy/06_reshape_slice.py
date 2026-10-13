import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
案例 06：reshape 与切片 — 数据形状变换
=============================================

核心概念：
  - reshape: 改变数组的维度结构，不改变数据
  - 切片 (slicing): 从数组中取子集，与 Java 的 substring 类似但更强大
  - 深度学习中频繁需要 reshape（如把图像展平、调整 batch 维度）

Java 类比：
  Java 切片: Arrays.copyOfRange(arr, 1, 4)
  NumPy:     arr[1:4]  或  arr[:, 1:4]

运行：python stage1_numpy/06_reshape_slice.py
"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    section("1. reshape — 改变形状不改变数据")

    arr = np.arange(12)  # [0,1,...,11]
    print(f"arr = {arr}")
    print(f"  shape = {arr.shape}")

    # reshape 成 3x4
    m34 = arr.reshape(3, 4)
    print(f"reshape(3,4) =\n{m34}")

    # reshape 成 2x2x3
    m223 = arr.reshape(2, 2, 3)
    print(f"reshape(2,2,3) =\n{m223}")

    # -1 表示"自动推断这一维"
    # 常用技巧: 展平为一维
    flat = m34.reshape(-1)
    print(f"reshape(-1) = {flat}")

    # 自动推断行数
    auto = arr.reshape(-1, 4)  # 12/4=3 行
    print(f"reshape(-1, 4) =\n{auto}")

    section("2. 切片基础 — 一维")

    arr = np.arange(10)
    print(f"arr = {arr}")

    # 基本语法: arr[start:stop:step]（不包含 stop）
    print(f"arr[2:5]     = {arr[2:5]}")  # [2 3 4]
    print(f"arr[:5]      = {arr[:5]}")  # [0 1 2 3 4]
    print(f"arr[5:]      = {arr[5:]}")  # [5 6 7 8 9]
    print(f"arr[::2]     = {arr[::2]}")  # [0 2 4 6 8]
    print(f"arr[1::2]    = {arr[1::2]}")  # [1 3 5 7 9]
    print(f"arr[::-1]    = {arr[::-1]}")  # 反转

    print("""
  Java 等价:
    Arrays.copyOfRange(arr, 2, 5)    → arr[2:5]
    但 Java 不支持 step、不支持负索引""")

    section("3. 切片基础 — 二维")

    matrix = np.arange(20).reshape(4, 5)
    print(f"matrix =\n{matrix}")

    # 语法: matrix[行切片, 列切片]
    print(f"matrix[1, 3]       = {matrix[1, 3]}")  # 标量: 第1行第3列
    print(f"matrix[1, :]       = {matrix[1, :]}")  # 第1行全部
    print(f"matrix[:, 2]       = {matrix[:, 2]}")  # 第2列全部
    print(f"matrix[1:3, :]    =\n{matrix[1:3, :]}")  # 第1-2行
    print(f"matrix[1:3, 2:4] =\n{matrix[1:3, 2:4]}")  # 第1-2行, 第2-3列

    # 取最后两行
    print(f"matrix[-2:, :]    =\n{matrix[-2:, :]}")

    section("4. 切片是视图而非拷贝 — 重要陷阱")

    arr = np.arange(5)
    print(f"arr = {arr}")

    # 切片返回的是视图(view)，不是新数组！
    sub = arr[1:4]
    print(f"sub = arr[1:4] = {sub}")

    # 修改 sub 会影响 arr
    sub[0] = 99
    print(f"修改 sub[0]=99 后:")
    print(f"  sub = {sub}")
    print(f"  arr = {arr}")  # arr[1] 也变成了 99！

    print("""
  *** Java 开发者注意 ***
  Java 里 Arrays.copyOfRange 返回的是新数组，修改不影响原数组
  NumPy 切片默认是视图，修改会影响原数组！

  如果需要独立拷贝，用 .copy():
    sub = arr[1:4].copy()""")

    section("5. 实战：图像数据形状变换")

    # 模拟一张 8x8 灰度图（64 个像素值）
    image = np.arange(64).reshape(8, 8)
    print(f"原始图像 (8x8):\n{image}")
    print(f"  shape = {image.shape}")

    # 展平为一维向量（送入全连接层时需要）
    flattened = image.reshape(-1)  # 或 image.flatten()
    print(f"展平后 shape = {flattened.shape}")  # (64,)

    # 模拟彩色图像: (高, 宽, 通道)
    color_image = np.random.default_rng(42).integers(0, 256, (4, 4, 3))
    print(f"\n彩色图像 shape = {color_image.shape}")  # (4, 4, 3)
    print(f"  4x4 像素，每像素 3 个通道 (RGB)")

    # 只取红色通道
    red = color_image[:, :, 0]
    print(f"红色通道 shape = {red.shape}")  # (4, 4)

    # 把 HWC 变成 CHW (PyTorch 需要的格式)
    chw = color_image.transpose(2, 0, 1)  # 把通道维度移到前面
    print(f"CHW 格式 shape = {chw.shape}")  # (3, 4, 4)

    section("6. 实战：batch 维度操作")

    # 模拟一个 batch: 32 个样本，每个 28x28 的灰度图（类似 MNIST）
    batch = np.random.default_rng(42).random((32, 28, 28))
    print(f"batch shape = {batch.shape}")  # (32, 28, 28)

    # 取第一个样本
    sample0 = batch[0]
    print(f"batch[0] shape = {sample0.shape}")  # (28, 28)

    # 取前 4 个样本
    first4 = batch[:4]
    print(f"batch[:4] shape = {first4.shape}")  # (4, 28, 28)

    # 展平每个样本为 784 维向量（送入全连接层）
    flattened_batch = batch.reshape(32, -1)  # 或 batch.reshape(batch.shape[0], -1)
    print(f"展平后 shape = {flattened_batch.shape}")  # (32, 784)

    section("7. np.newaxis — 增加维度")

    arr = np.array([1, 2, 3])
    print(f"arr shape = {arr.shape}")  # (3,)

    # 增加一个维度，变成列向量
    col = arr[:, np.newaxis]
    print(f"arr[:, np.newaxis] shape = {col.shape}")  # (3, 1)
    print(f"  内容 =\n{col}")

    # 变成行向量
    row = arr[np.newaxis, :]
    print(f"arr[np.newaxis, :] shape = {row.shape}")  # (1, 3)

    # 在深度学习中常用: 把单样本变成 batch_size=1
    # (3,) → (1, 3) 模拟一个 batch
    print(f"\n深度学习用途: (特征,) → (1, 特征) 模拟 batch")

    print("\n[OK] 案例 06 完成 - reshape 和切片是数据形状操作的核心")
    print("     关键认知: 切片是视图不是拷贝; reshape(-1) 是万能展平工具")


if __name__ == "__main__":
    main()
