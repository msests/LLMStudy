## 概述

传统注意力机制（如Transformer中的Scaled Dot-Product Attention）的计算复杂度为$O(N^2)$（$N$为序列长度），限制了其在长序列任务中的应用。**Linear Attention**通过数学变换将复杂度降低到$O(N)$，同时保持注意力机制的核心特性。其核心思想是**解耦Softmax计算**并利用矩阵乘法的结合律优化计算顺序。

## 原理细节

### 标准注意力机制回顾

标准注意力计算流程如下：

1. 输入矩阵：$Q \in \mathbb{R}^{N \times d}, K \in \mathbb{R}^{N \times d}, V \in \mathbb{R}^{N \times d}$
2. 计算注意力矩阵：
$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \in \mathbb{R}^{N \times N}
$$
3. 输出：
$$O = AV \in \mathbb{R}^{N \times d}$$

计算复杂度主要来自$QK^T$（$O(N^2d)$）和softmax操作。

### Linear Attention的数学变换

关键步骤：将softmax分解为两个独立的计算过程。

1. **分解Softmax**：
$$
\text{softmax}(QK^T)V = \frac{\exp(QK^T)}{\sum_{i=1}^N \exp(QK_i^T)}V
$$
   改写为：
$$
O = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1}_N)}
$$
   其中$\phi(\cdot)$为核函数（如指数函数$\phi(x)=\exp(x/\sqrt{d})$）

2. **改变计算顺序**：

   利用矩阵结合律：
$$
O = \frac{\phi(Q) \cdot [\phi(K)^T V]}{\phi(Q) \cdot [\phi(K)^T \mathbf{1}_N]}
$$
3. **复杂度分析**：
   - $\phi(K)^T V \in \mathbb{R}^{d \times d}$ 的计算复杂度为$O(Nd^2)$
   - 最终乘法$\phi(Q) \cdot [\phi(K)^T V]$的复杂度为$O(Nd^2)$
   - 总体复杂度从$O(N^2d)$降为$O(Nd^2)$

### 核函数选择

常见核函数实现线性化：

| 核函数类型 | 表达式 | 特性 |
|---------|--------|-----|
| 指数核   | $\phi(x)=\exp(x/\sqrt{d})$ | 严格等价于标准注意力 |
| 多项式核 | $\phi(x)=(x/\sqrt{d}+1)^k$ | 可调节多项式阶数$k$ |
| 线性核   | $\phi(x)=\text{elu}(x)+1$ | 完全消除softmax |

### 位置编码整合

由于改变了注意力计算方式，需调整位置编码：
- 使用**相对位置编码**：$Q_i^T K_j \rightarrow Q_i^T K_j + b_{i-j}$
- 或**核函数扩展**：$\phi([x_i; p_i])$（拼接位置编码）

## 优点和缺点

### 优点

- **计算高效**
	- 复杂度$O(Nd^2)$ vs 标准$O(N^2d)$，尤其适合长序列（如DNA分析）。
- **内存优化**
	- 无需存储$N \times N$注意力矩阵，只需要存储$d \times d$的矩阵。
- **灵活性**
	- 可通过不同核函数控制注意力模式，在准确性和高效性之间做选择。

### 缺点

- **核函数敏感**
	- 性能高度依赖核函数的选择。
	- 核函数可能无法精确匹配标准注意力分布，存在一定的误差。
- **局部注意力退化**
	- 某些核函数可能削弱局部依赖捕捉能力。

## 等价性分析

### 结论

**当且仅当使用指数核函数时**，改变计算顺序不会改变最终结果，此时Linear Attention与标准注意力严格等价。对于其他核函数，计算顺序的改变会引入近似误差。

### 严格等价性证明

#### 数学推导

设核函数为$\phi(x) = \exp(x/\sqrt{d})$，验证两种计算路径的等价性：

**路径1（标准注意力）**：
$$
O_1 = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V = \frac{\exp(QK^T/\sqrt{d})}{\text{rowsum}(\exp(QK^T/\sqrt{d}))}V
$$

**路径2（Linear Attention）**：
$$
O_2 = \frac{\phi(Q)[\phi(K)^T V]}{\phi(Q)[\phi(K)^T \mathbf{1}_N]}
$$

**展开计算**：
1. 分子部分：
$$
\phi(Q)[\phi(K)^T V] = \exp(Q/\sqrt{d}) \cdot [\exp(K/\sqrt{d})^T V] 
= \exp(QK^T/\sqrt{d})V
$$

2. 分母部分：
$$
\phi(Q)[\phi(K)^T \mathbf{1}_N] = \exp(Q/\sqrt{d}) \cdot [\exp(K/\sqrt{d})^T \mathbf{1}_N] 
= \text{rowsum}(\exp(QK^T/\sqrt{d}))
$$

3. 最终结果：
$$
O_2 = \frac{\exp(QK^T/\sqrt{d})V}{\text{rowsum}(\exp(QK^T/\sqrt{d}))} = O_1
$$

#### 几何解释


### 非指数核情况下的近似误差

#### 数学分析

设使用一般核函数$\phi(\cdot)$，定义误差项：
$$
\Delta = \left\| \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1}_N)} - \text{softmax}(QK^T)V \right\|_F
$$

#### 误差来源

1. **核函数表达能力限制**  
   当$\phi(Q)\phi(K)^T \neq \exp(QK^T/\sqrt{d})$时，无法保持双线性形式。

2. **归一化偏移**  
   分母$\phi(K)^T \mathbf{1}_N$与标准softmax的rowsum存在几何差异。

#### 误差量化

通过泰勒展开分析多项式核$\phi(x)=(x/\sqrt{d}+1)^k$的近似误差：
$$
\phi(Q)\phi(K)^T = \left(\frac{QK^T}{d} + \frac{Q+K}{\sqrt{d}} + 1 \right)^k
$$
与标准注意力核$\exp(QK^T/\sqrt{d})$的KL散度：
$$
D_{KL} \propto k^2 \cdot \text{Tr}(QK^T)^2/d^2
$$