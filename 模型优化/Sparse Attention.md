## 概述

Sparse Attention 是传统注意力机制（如Transformer中的自注意力）的一种优化变体，旨在**降低计算复杂度和内存消耗**。传统自注意力计算所有输入位置对之间的关联性，复杂度为$O(n^2)$（n为序列长度）。Sparse Attention通过**限制注意力作用的范围**，将复杂度降低到$O(n \log n)$或$O(n)$，使其能够处理更长序列（如文档级文本或高分辨率图像）。

## 原理

### 标准注意力回顾

标准自注意力计算过程：  

对于输入序列$X \in \mathbb{R}^{n \times d}$，计算$Q,K,V$：  
$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V  
$$
注意力权重矩阵$A \in \mathbb{R}^{n \times n}$：  
$$
A = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)  
$$
输出为$AV$。计算复杂度为$O(n^2 d)$，内存占用为$O(n^2)$。

### 稀疏注意力核心思想

通过**结构化稀疏模式**替代全连接注意力，定义稀疏矩阵$M \in \{0,1\}^{n \times n}$，仅当$M_{ij}=1$时计算$i$对$j$的注意力权重：  
$$
A_{sparse} = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \odot M \right)  
$$
其中$\odot$为逐元素乘法。

### 主要实现方式

#### a) 局部注意力（Local Attention）

- **滑动窗口**：每个位置仅关注前后$k$个位置（窗口大小$2k+1$）  
$$
M_{ij} = 1 \quad \text{当且仅当} \quad |i-j| \leq k  
$$  
  复杂度$O(nk)$，适合局部相关性强的任务（如文本、语音）。

- **块状局部注意力**：将序列分为固定块，块内全连接（如ImageGPT）。

#### b) 全局注意力（Global Attention）

- 指定少量**全局标记**，这些标记与所有位置交互：  
  $$
  M_{ij} = 1 \quad \text{若} \quad i \in S_{global} \ \text{或} \ j \in S_{global}  
  $$  
  复杂度$O(n|S_{global}|)$，常用于捕获文档级信息。

#### c) 随机注意力（Random Attention）

- 每个位置随机选择$r$个位置建立连接：  
  $$
  M_{ij} = 1 \quad \text{若} \quad j \in \text{RandomSample}([1,n], r)  
  $$  
  复杂度$O(nr)$，通过随机性近似全连接效果（BigBird使用）。

#### d) 下采样注意力（Downsampling Attention）

- 通过池化或卷积降低序列长度：  
  - 对$K, V$进行下采样（如Linformer投影到低维）  
  - 复杂度从$O(n^2)$降为$O(nk)$（$k \ll n$）

### 混合策略

实际模型常组合多种稀疏模式：  
- **Longformer**：滑动窗口 + 任务相关全局注意力  
- **BigBird**：局部 + 全局 + 随机注意力（理论可近似全连接）  
- **Sparse Transformer**：跨步局部注意力（Strided Attention）

## 稀疏矩阵的存储形式

在 Sparse Attention 中，核心是**稀疏注意力矩阵**的高效存储（$K,Q,V$仍然需要用常规的存储方式）。以下是常见的存储策略：

### 稀疏矩阵格式

- **Coordinate Format (COO)**  
  存储非零元素坐标`(i,j)`及其值，适用于高度稀疏且非零元素分布随机的场景（如Random Attention）。  
```python
  # 示例：存储3个非零元素
  indices = [[0,0], [1,2], [2,3]]  # 坐标
  values  = [0.8, 0.5, 0.3]        # 对应值
```

- **Compressed Sparse Row (CSR)**  
  通过行偏移指针压缩存储，适合行方向稀疏性显著的情况（如Local Attention的滑动窗口）：
```python
    data = [0.8, 0.5, 0.3]    # 非零值
    indices = [0, 2, 3]        # 列索引
    indptr = [0, 1, 2, 3]      # 行偏移指针
```

- **Block Sparse Format**  
  将矩阵划分为块（如4x4），仅存储含非零元素的块（如Sparse Transformer的固定模式）：
```python
    # 块稀疏布局示例
    block_mask = [
        [1, 0, 1],  # 每个1代表一个非零块
        [0, 1, 0],
        [1, 0, 1]
    ]
```

#### 框架支持

PyTorch通过 `torch.sparse` 模块支持 COO/CSR 格式：

```python
    sparse_A = torch.sparse_coo_tensor(indices, values, size=(n, n))
    output = torch.sparse.mm(sparse_A, dense_B)
```

NVIDIA支持块稀疏格式

```cpp
    // CUDA示例：块稀疏矩阵乘法
    cusparseLtMatmul(..., CUSPARSELT_MATMUL_ALG_DEFAULT, &sparsity);
```

### 结构化稀疏的隐式存储

对于**固定模式**（如Local Attention的滑动窗口），可通过计算规则动态生成掩码，无需显式存储矩阵：
```python
# 滑动窗口的隐式掩码生成
def generate_local_mask(seq_len, window_size):
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1
    return mask
```

## 优点和缺点

### 优点

- **计算效率**
	- 复杂度从$O(n^2)$降至线性或亚线性  
- **内存优化**
	- 减少注意力得分矩阵的显存占用，支持更长序列（如4096+ tokens）  
- **可扩展性**
	- 适配图像、视频等高维数据  
- **理论保证**
	- 部分方法（如BigBird）可保持Universal Approximator性质

### 缺点

- **信息损失**
	- 可能忽略长距离依赖（需依赖全局注意力补偿）。 
- **实现复杂性**
	- 需定制CUDA内核优化稀疏计算（如块稀疏注意力）。
- **任务敏感性**
	- 需人工设计稀疏模式（如局部窗口大小）。
- **近似误差**
	- 随机或下采样方法可能引入噪声。

