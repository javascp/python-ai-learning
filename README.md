# python-ai-learning — 从 Java 到 AI 原理

> 面向 Java 开发者的 Python AI 学习项目。用 Java 思维理解 AI，从 NumPy 到 Transformer，每个知识点都有代码 + Java 类比 + 练习。

如果你是 Java 开发者，想学 AI 但不知道从哪里开始——这个项目就是为你做的。

## 为什么做这个项目

市面上的 AI 教程要么太偏理论（满纸数学公式），要么太偏工程（直接调 API 不讲原理）。这个项目走中间路线：

- **不回避原理**，但每个概念都用代码讲清楚
- **不假设你懂数学**，用到时按需补齐
- **不假设你懂 Python**，用 Java 概念做锚点快速建立映射
- **每个案例都能跑**，不是截图式教程

## 学习路径

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1. NumPy 工具链 | ndarray、向量化、广播、矩阵运算 | ✅ 已完成 |
| 2. 数学基础 | 线性代数、微积分、概率统计 | 🔜 进行中 |
| 3. 机器学习 | 线性回归、梯度下降、分类 | 📋 规划中 |
| 4. 深度学习 | 神经网络、反向传播、CNN/RNN | 📋 规划中 |
| 5. Transformer | 注意力机制、大语言模型 | 📋 规划中 |

## 目录结构

```
python-ai-learning/
├── run_all.py                ← 一键运行所有案例
├── requirements.txt          ← 依赖清单
│
├── stage1_numpy/             ← 第 1 阶段：Python 科学计算工具链
│   ├── 01_array_basics.py        ndarray 基础，shape/dtype/ndim
│   ├── 02_vectorized_ops.py      向量化运算，50-100x 加速原理
│   ├── 03_broadcasting.py       广播机制，神经网络偏置加法基础
│   ├── 04_boolean_indexing.py   布尔索引，数据筛选与清洗
│   ├── 05_matrix_ops.py         矩阵运算，y = W @ x + b 神经网络核心
│   ├── 06_reshape_slice.py      形状变换，图像数据/批量处理
│   └── 07_random_stats.py       随机数与统计，权重初始化
│
├── stage2_math/              ← 第 2 阶段：数学基础（按需补齐）
├── stage3_ml/                ← 第 3 阶段：机器学习核心原理
├── stage4_deep_learning/     ← 第 4 阶段：深度学习原理
└── stage5_transformer/       ← 第 5 阶段：Transformer 与大语言模型
```

## 快速开始

```bash
# 克隆项目
git clone https://github.com/javascp/python-ai-learning.git
cd python-ai-learning

# 安装依赖
pip install -r requirements.txt

# 运行单个案例
python stage1_numpy/01_array_basics.py

# 一键运行整个阶段
python run_all.py stage1

# 运行指定案例（如第 5 个：矩阵运算）
python run_all.py stage1 5
```

## 每个案例的结构

```
┌─────────────────────────────────────────┐
│  概念讲解    顶部注释说明核心知识点        │
│  Java 类比   关键概念标注 Java 对应物      │
│  代码演示    可运行、有注释、有输出预期     │
│  练习题      TODO 标记，补全后运行验证     │
└─────────────────────────────────────────┘
```

以案例 05「矩阵运算」为例，你会在注释里看到这样的 Java 对比：

```python
# Java 三层 for 循环手写矩阵乘法:
#     for (int i=0; i<M; i++)
#         for (int j=0; j<N; j++)
#             for (int k=0; k<K; k++)
#                 C[i][j] += A[i][k] * B[k][j];
# NumPy: C = A @ B
```

## 谁适合用这个项目

- **Java 开发者**，想转 AI 方向但缺乏起点
- **有编程基础的人**，想从原理理解 AI 而非调包
- **正在学 AI 的人**，想要可运行、有注释的参考代码

## 技术栈

- Python 3.10+
- NumPy — 数值计算基础
- Pandas — 数据处理（后续阶段）
- Matplotlib — 可视化（后续阶段）

## 项目规划

- [x] Stage 1: NumPy 工具链（7 个案例）
- [ ] Stage 2: 数学基础
- [ ] Stage 3: 机器学习核心原理
- [ ] Stage 4: 深度学习原理
- [ ] Stage 5: Transformer 与大语言模型

## License

MIT License — 自由使用、修改、分享。

## Star History

如果这个项目对你有帮助，点个 Star 让更多人看到。

如果这个项目对你有帮助，欢迎 Star ⭐ 支持！有问题或建议可以提 [Issue](https://github.com/javascp/python-ai-learning/issues)。
