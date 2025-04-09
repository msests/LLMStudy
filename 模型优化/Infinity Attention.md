## 概述

Infinity Attention 是一种针对长序列处理的注意力机制改进方法，旨在解决传统Transformer模型中自注意力（Self-Attention）在长序列场景下计算复杂度高的问题。其核心思想是通过**分块递归**和**局部-全局注意力融合**，在保持全局上下文感知能力的同时，显著降低计算开销。

## 技术细节

### 核心原理

传统自注意力复杂度为 $O(n^2)$（n为序列长度），而Infinity Attention通过以下步骤实现 $O(n)$ 复杂度：

1. **序列分块**  
   将输入序列 $X \in \mathbb{R}^{n \times d}$ 分割为 $m$ 个块，每块长度 $l = n/m$：
$$
X = [X_1, X_2, ..., X_m], \quad X_i \in \mathbb{R}^{l \times d}
$$

2. **局部注意力**  
   在每个块内计算标准自注意力：
$$
\text{LocalAttn}(X_i) = \text{Softmax}\left(\frac{X_i W^Q (X_i W^K)^T}{\sqrt{d}}\right) X_i W^V
$$

3. **全局记忆单元**  
   维护一个可学习的全局记忆矩阵 $M \in \mathbb{R}^{k \times d}$（k为固定大小），通过跨块交互更新：
$$
M_{new} = \text{LayerNorm}(M + \text{Attn}(M, X))
$$

4. **递归融合**  
   将全局记忆与前一块的输出结合，传递到下一块：
$$
\tilde{X}_i = \text{LocalAttn}(X_i) + \alpha \cdot \text{Attn}(X_i, M)
$$

### 关键技术

- **局部敏感哈希（LSH）**：加速相似度计算，对查询-键对进行哈希分桶
- **动态上下文选择**：通过门控机制决定局部/全局注意力的权重 $\alpha$
- **梯度检查点**：减少反向传播时的内存占用

### 数学形式化

完整计算流程可表示为：
$$
\begin{aligned}
Q & = XW^Q, \quad K = XW^K, \quad V = XW^V \\
\hat{A}_{ij} & = \begin{cases} 
\frac{Q_i K_j^T}{\sqrt{d}} & \text{if } \lfloor i/l \rfloor = \lfloor j/l \rfloor \\
g(Q_i, K_j) & \text{otherwise}
\end{cases} \\
\text{Output} & = \text{Softmax}(\hat{A})V
\end{aligned}
$$
其中 $g(\cdot)$ 为基于LSH的近似相似度函数。

## 优点与缺点

### 优点

1. **计算效率**：复杂度从 $O(n^2)$ 降为 $O(n)$，适合处理超长序列（如DNA分析）
2. **内存优化**：峰值内存消耗降低约40%（以序列长度10k为例）
3. **保持全局性**：通过记忆单元保留跨块依赖关系

### 缺点

1. **近似误差**：LSH和分块策略可能损失部分注意力精度
2. **实现复杂度**：需要精细设计分块大小和记忆更新策略
3. **局部性假设**：对需要密集全局交互的任务（如机器翻译）可能不适用
