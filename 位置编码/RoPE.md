## 概述

旋转位置编码(Rotary Position Embedding, RoPE)是一种将位置信息融入Transformer自注意力机制的方法。其核心思想是通过**旋转矩阵**对查询(Query)和键(Key)向量进行变换，使注意力分数自然包含相对位置信息。该方法被广泛应用于LLaMA、ChatGLM等大语言模型。

## 数学原理

### 基本形式

对于位置$m$的向量$\boldsymbol{x}_m \in \mathbb{R}^d$，RoPE将其映射为复数形式：
$$
\boldsymbol{x}_m^{(c)} = \boldsymbol{x}_m e^{im\theta}
$$
其中$\theta$是预设角度参数。在实数空间中等价于：
$$
\boldsymbol{R}_m = \begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}
$$

### 高维扩展

将$d$维向量分为$d/2$组二维子空间，每组应用独立旋转矩阵。对于第$t$组：
$$
\boldsymbol{R}_{\Theta,t}^m = \begin{bmatrix}
\cos m\theta_t & -\sin m\theta_t \\
\sin m\theta_t & \cos m\theta_t
\end{bmatrix}, \quad 
\theta_t = 10000^{-2t/d}
$$
其中$\theta_t$按几何级数设置，与Transformer原版位置编码一致。

### 自注意力应用

给定查询$\boldsymbol{q}$和键$\boldsymbol{k}$，RoPE计算为：
$$
\begin{aligned}
\boldsymbol{q}_m &= \boldsymbol{R}_{\Theta}^m \boldsymbol{W}_q \boldsymbol{x}_m \\
\boldsymbol{k}_n &= \boldsymbol{R}_{\Theta}^n \boldsymbol{W}_k \boldsymbol{x}_n \\
\end{aligned}
$$
注意力分数计算时：
$$
\boldsymbol{q}_m^T \boldsymbol{k}_n = (\boldsymbol{R}_{\Theta}^m \boldsymbol{q})^T (\boldsymbol{R}_{\Theta}^n \boldsymbol{k}) = \boldsymbol{q}^T \boldsymbol{R}_{\Theta}^{n-m} \boldsymbol{k}
$$
推导过程中利用旋转矩阵正交性$\boldsymbol{R}^T_m \boldsymbol{R}_n = \boldsymbol{R}_{n-m}$

### 实现优化

实际实现通过以下等价计算避免显式矩阵乘法：
$$
\begin{bmatrix}
q_j \cos m\theta_j - q_{j+1} \sin m\theta_j \\
q_j \sin m\theta_j + q_{j+1} \cos m\theta_j
\end{bmatrix}, \quad \forall j=0,2,4,...,d-2
$$

## 优点和缺点

### 优点

- **良好的外推性**
	- 通过旋转操作的自然延拓，比绝对位置编码更适应长文本
- **保持相对位置信息**
	- 注意力分数仅依赖相对位置$m-n$
- **线性可加性**
	- 满足$\boldsymbol{R}_m \boldsymbol{R}_n = \boldsymbol{R}_{m+n}$，即旋转角度可以叠加。
- **兼容高效计算**
	- 可与FlashAttention等优化技术结合

### 缺点

- **计算复杂度略高**
	- 相比原始位置编码增加约15%计算量
- **高频信息损失**
	- 高旋转频率可能导致数值不稳定
- **参数敏感性**
	- 旋转角度的基频选择影响性能