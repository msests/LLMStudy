## 1. 概述

**Adapter Tuning**（适配器微调）是一种参数高效的微调方法，最早由 Houlsby 等人在 2019 年提出。该方法通过在每层 Transformer 中插入小型的可训练模块（adapter），仅训练这些模块而保持原始模型参数冻结，实现低成本的模型适配和迁移学习。

## 2. 原理

Adapter 的核心思想是：**在冻结原始模型的基础上，引入额外的可训练模块来学习特定任务的表示**。

### 2.1 Adapter模块

Adapter模块可以插入在Transformer的任意地方，通常实践中会插入在两个Normalization层之后。

![[AdapterPosition.drawio.svg]]

一个标准的 Adapter 模块包括：
1. **降维层（Down Projection）**：将输入从高维压缩到一个低维空间（如从 768 → 64）。
2. **非线性激活函数**（通常是 ReLU 或 GELU）。
3. **升维层（Up Projection）**：将低维表示恢复到原始维度。
4. **残差连接**：将 Adapter 的输出与原始输入相加。

数学形式：
$$Adapter(x) = x + W_{up}(ReLU(W_{down}(x)))$$
由于仅训练少量 adapter 参数，原模型保持不变，因此多任务共享主干模型，任务间互不干扰。

### 2.2 参数估计

单个Adapter的参数量：
$$\text{Params\_per\_adapter} = 2 × d_{model} × r + 2 × r (bias)$$
其中：
- $d_{model}$是输入的维度。
- $r$是降维参数。

假设每层插入两个Adapter的话，那么模型中总共新增的参数量为：
$$\text{Total\_params} = \text{N\_layers} \times 2 \times (2 \times d_{model} \times r + 2 \times r)$$

## 3. 实现

以下为基于 HuggingFace Transformers 框架的 Adapter Tuning 实现流程简述：

### 3.1 加载预训练模型并添加 adapter

```python
from transformers import AutoAdapterModel, AutoTokenizer

model = AutoAdapterModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 添加 adapter
model.add_adapter("my-task")
model.train_adapter("my-task")
```

### 3.2 训练流程

将模型与适配器一起训练：

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(output_dir="output", num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=train_data)

trainer.train()
```

### 3.3 推理流程

加载 adapter 并进行推理：

```python
model.load_adapter("my-task", with_head=False)
model.set_active_adapters("my-task")

outputs = model(**tokenizer("Example input", return_tensors="pt"))
```

## 4. 优点与缺点

### 4.1 优点

- **参数高效**
	- 通常只训练 1%-5% 的参数。
- **多任务友好**
	- 每个任务一个 adapter，不干扰原始模型。
- **便于部署**
	- 主模型参数共享，adapter 可热插拔。
- **快速训练**
	- 更快的训练速度和更低的显存占用。

### 4.2 缺点

- **表达能力受限**
	- 小模块可能无法完全适应复杂任务。
- **工程成本**
	- 需额外适配模型结构或使用特定库。
- **缺乏统一标准**
	- adapter 尺寸、插入位置等需调参。
- **不适用于所有模型**
	- 例如部分非 Transformer 模型适配困难。
