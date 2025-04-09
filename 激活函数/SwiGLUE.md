## 概述

SwiGLU（Swish-Gated Linear Unit）是一种结合了**Swish激活函数** 和**门控线性单元（GLU）** 结构的神经网络激活函数。首次在Transformer模型（如PaLM、LLaMA）中被广泛采用，通过引入可学习的门控机制增强非线性表达能力。其核心思想是通过门控权重动态调节信息流，同时利用Swish的平滑梯度特性缓解梯度消失问题。

## 数学原理与特性

### 基础组件

#### Gated Linear Unit (GLU)

GLU的原始形式定义为：
$$
\text{GLU}(x) = x \otimes \sigma(g(x))
$$
其中：
- $x \in \mathbb{R}^d$ 是输入向量
- $g(x) = Wx + b$ 是线性变换（$W \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$）
- $\sigma$ 是Sigmoid函数，输出门控权重
- $\otimes$ 表示逐元素乘积

#### Swish 激活函数

Swish定义为：
$$
\text{Swish}(x) = x \cdot \sigma(\beta x)
$$
其中 $\beta$ 是可学习参数或固定值。当 $\beta \to 0$ 时接近线性函数，$\beta \to \infty$ 时接近ReLU。

### SwiGLU 的数学形式

SwiGLU将GLU中的Sigmoid替换为Swish函数：
$$
\text{SwiGLU}(x) = (xW + b) \otimes \text{Swish}(xV + c)
$$
展开后：
$$
\text{SwiGLU}(x) = (xW + b) \otimes \left( (xV + c) \cdot \sigma(\beta (xV + c)) \right)
$$
其中：
- $W, V \in \mathbb{R}^{d \times h}$ 是投影矩阵（通常 $h = \frac{2d}{3}$）
- $b, c \in \mathbb{R}^h$ 是偏置项
- $\beta$ 可设为1或作为可学习参数

### 特性分析

1. **非线性增强**：双重非线性（Swish和门控乘积）提升模型表达能力。
2. **梯度平滑性**：Swish连续可导且非饱和，缓解梯度消失。
3. **动态门控**：通过 $xV + c$ 生成输入相关的门控权重。
4. **参数效率**：投影维度 $h$ 通常小于输入维度 $d$（如 $h=2d/3$）。

## 优点与缺点

### 优点

- **性能优越**
	- 在语言模型中表现优于ReLU/GELU，尤其在深层网络。
- **梯度稳定性**
	- Swish的平滑性减少训练中的梯度突变。
- **灵活门控**
	- 动态调节信息流，增强模型适应性。

### 缺点

- **计算开销**
	- 相比标准激活函数，增加约30%的计算量（因双投影和逐元素乘）。
- **参数量增加**
	- 引入额外参数 $V$ 和 $c$（但可通过降低 $h$ 缓解）。