## 训练过程

### 预处理

- **分词**：将输入Prompts分解为Token序列。

### 输入处理

- **词嵌入**：将输入序列 $X = (x_1, x_2, ..., x_n)$ 映射为 $d_{model}$ 维向量 $E = (\vec e_1, \vec e_2, ..., \vec e_n)$
- **位置编码**：加入位置信息，可以采用：
	- 正余弦位置编码
	- 相对距离编码
	- RoPE
	- ALiBi

### 编码器堆叠

每层编码器包含：

1. **多头注意力**：
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
   多头拼接：$\text{MultiHead} = \text{Concat}(head_1,...,head_h)W^O$
   
2. **前馈网络**：
$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

3. **残差连接与层归一化**：
$$
x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

### 解码器结构

解码器额外包含：
- **掩码多头注意力**：防止看到未来信息
- **编码-解码注意力**：接收编码器输出作为K,V

### 训练流程

1. 前向传播：计算预测分布 $P(y_t|y_{<t}, X)$
2. 损失计算：交叉熵损失
$$
\mathcal{L} = -\sum_{t=1}^T \log P(y_t^*|y_{<t}, X)
$$
3. 反向传播：通过梯度下降优化参数
4. 优化技巧：
   - 学习率预热（Warmup）
   - 标签平滑（Label Smoothing）
   - Dropout 正则化

## 推理过程

### 自回归生成

采用逐步生成策略：
1. 初始化输入：起始符 `<sos>`
2. 循环生成直到出现 `<eos>`：
$$
y_t = \arg\max_{w} P(w|y_{<t}, X)
$$

### 2. 关键技术
- **Beam Search**：维护k个候选序列
  ```python
  beams = [([], 0)]  # (sequence, score)
  for t in range(max_len):
      new_beams = []
      for seq, score in beams:
          logits = model(seq)
          top_k = logits.topk(k)
          for token, log_prob in top_k:
              new_beams.append((seq+[token], score+log_prob))
      beams = sorted(new_beams, key=lambda x: x[1])[:k]
```

- **采样策略**：
    
    - 贪心搜索（Greedy）
        
    - Top-k采样
        
    - 温度缩放（Temperature Scaling）
        

### 3. 优化技术

- **KV缓存**：缓存先前计算的Key/Value矩阵
    
- **并行计算**：对beam search进行批处理
    
- **长度惩罚**：调整长序列的得分
    

## 三、核心差异对比

|特性|训练阶段|推理阶段|
|---|---|---|
|输入长度|固定长度（Padding/Mask）|动态增长（自回归）|
|注意力掩码|全可见（编码器）|因果掩码（解码器）|
|并行性|完全并行|序列依赖（需缓存状态）|
|目标输出|已知完整序列|逐步生成|
|计算资源|大批量GPU并行|低延迟优化（如ONNX）|