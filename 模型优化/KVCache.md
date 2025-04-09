## 概述

KVCache（Key-Value Cache）是Transformer架构在**自回归生成任务**中用于加速推理的关键技术。在大型语言模型（LLM）的解码阶段（如文本生成），模型需要逐步生成每个token，而KVCache通过缓存历史token的键（Key）和值（Value）矩阵，避免了重复计算，显著降低了计算复杂度。

## 技术细节

### 自注意力机制与KVCache原理

在Transformer的自注意力中，对于输入序列$X \in \mathbb{R}^{n \times d}$，通过线性变换得到Query、Key、Value矩阵：
$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$
注意力得分为：
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**解码阶段的特殊需求**：生成第$t$个token时，只需要计算当前token的$q_t \in \mathbb{R}^{1 \times d}$，但需要与所有历史token的$k_{1:t}$和$v_{1:t}$交互。KVCache通过缓存历史$K_{1:t-1}$和$V_{1:t-1}$，使得：
$$
K_{1:t} = [K_{1:t-1}; k_t], \quad V_{1:t} = [V_{1:t-1}; v_t]
$$

### 实现细节

1. **缓存初始化**
	1. 在首个生成步骤（prompt处理阶段），计算并存储所有输入token的$K,V$。
2. **增量更新**：后续每生成一个新token，仅计算当前token的$k_t, v_t$，并追加到缓存。
3. **内存布局**：通常维护两个张量`cache_k`和`cache_v`，形状为`[batch_size, num_heads, seq_len, head_dim]`

### 计算复杂度分析
| 场景       | 计算复杂度     | 内存复杂度   |
| -------- | --------- | ------- |
| 无KVCache | $O(t^2d)$ | $O(td)$ |
| 有KVCache | $O(td)$   | $O(td)$ |
*其中$t$为序列长度，$d$为隐藏层维度*

### 内存占用定量分析

假设：
- 批次大小 $b$
- 注意力头数 $h$
- 每头维度 $d_h$
- 层数 $L$
- 序列长度 $s$

每层的KVCache内存占用为：
$$
\text{每层内存} = 2 \times b \times s \times h \times d_h \quad (\text{bytes})
$$
总内存占用量为：
$$
\text{总内存} = L \times 2 \times b \times s \times h \times d_h \times \text{dtype\_size}
$$

## 优点与缺点

### 优点

- **显著降低计算量**
	- 将自注意力复杂度从$O(n^2)$降为$O(n)$。
- **支持长序列生成**
	- 与全量重计算相比，内存增长更平缓。

### 缺点

- **内存消耗大**
	- 对于175B参数的模型，生成1024 token需约1.2GB显存（fp16）
- **缓存管理复杂**
	- 需要处理不同序列长度的填充（padding）
- **硬件限制**
	- 当$s$超过GPU显存容量时，需要特殊处理（如分块）

## 4. 优化方向（扩展）
- **动态缓存裁剪**：基于注意力分数剪枝不重要token的KV
- **量化压缩**：使用8-bit或4-bit存储KV
- **分页缓存**：类似vLLM的PagedAttention技术
- **选择性更新**：跳过对后续生成影响小的token的KV存储