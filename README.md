# python-ai-learning — 从 Java 到 AI 原理

> 面向 Java 开发者的 Python AI 学习项目。每个阶段独立目录，每个案例可单独运行。

## 目录结构

```
python-ai-learning/
├── README.md                 ← 本文件
├── run_all.py                ← 一键运行所有案例
├── requirements.txt          ← 依赖清单
│
├── stage1_numpy/             ← 第 1 阶段：Python 科学计算工具链
│   ├── 01_array_basics.py
│   ├── 02_vectorized_ops.py
│   ├── 03_broadcasting.py
│   ├── 04_boolean_indexing.py
│   ├── 05_matrix_ops.py
│   ├── 06_reshape_slice.py
│   └── 07_performance.py
│
├── stage2_math/              ← 第 2 阶段：数学基础（按需补齐）
│
├── stage3_ml/                ← 第 3 阶段：机器学习核心原理
│
├── stage4_deep_learning/     ← 第 4 阶段：深度学习原理
│
└── stage5_transformer/       ← 第 5 阶段：Transformer 与大语言模型
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单个案例
python stage1_numpy/01_array_basics.py

# 一键运行所有案例
python run_all.py
```

## 学习路径

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1. NumPy 工具链 | ndarray、向量化、广播、矩阵运算 | 已完成 |
| 2. 数学基础 | 线性代数、微积分、概率统计 | 待开发 |
| 3. 机器学习 | 线性回归、梯度下降、分类 | 待开发 |
| 4. 深度学习 | 神经网络、反向传播、CNN/RNN | 待开发 |
| 5. Transformer | 注意力机制、大语言模型 | 待开发 |

## 学习方式

每个案例文件结构：
- **概念讲解**：顶部注释说明本案例的核心知识点
- **Java 类比**：关键概念标注 Java 对应物
- **代码演示**：可运行、有注释、有输出预期
- **练习题**：TODO 标记，补全后运行验证
