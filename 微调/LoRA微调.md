## 1. 概述

LoRA（Low-Rank Adaptation）是一种高效的参数微调方法，旨在在冻结大模型大部分参数的前提下，通过引入低秩矩阵的训练模块来实现模型的适配与微调。LoRA 通常用于大语言模型（LLMs）等大型预训练模型的下游任务微调中，具有低计算成本和低内存开销的优势。

## 2. 原理

在传统微调中，整个模型的权重都会参与训练，资源开销大。而 LoRA 的核心思想是：**在保持原始预训练模型权重冻结的情况下，仅在部分关键层（如注意力层）中插入一个低秩的参数变换模块进行训练**。

### 2.1 数学表达

以一个线性变换为例：

原始模型中的变换为：
$$y = W_0x$$

LoRA 引入一个可训练的低秩变换：
$$y = W_0x + \Delta Wx = W_0x + BAx$$

其中：

- $W₀ \in \mathbb{R}^{d \times d}$ 是冻结的原始权重
- $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$是可训练的低秩矩阵（秩 $r \ll d$）
- $\Delta W = BA$ 表示对权重的增量

这种方式使得训练参数从 $d^2$ 降低到 $2dr$。

### 2.2 参数估计

**LoRA参数数量估计公式：**
$$LoRA参数量=r⋅(d_{in}+d_{out})$$

- $r$：低秩维度（LoRA 的秩）。
- $d_{in}$​：降维前的维度。
- $d_{out}$​：升维后的维度。

假如每层

## 3. 实现

以 Hugging Face 的 `peft` 和 `transformers` 框架为例，LoRA 实现流程如下：

### 3.1 加载模型并应用 LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 只在注意力层插入LoRA
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
```

### 3.2 训练与保存

LoRA 训练过程与普通 Transformers 模型一致，但训练参数更少，训练更快。训练完成后，只需保存 LoRA 模块即可：

```python
model.save_pretrained("output/lora_model")
```

## 4. 优点和缺点

### 4.1 优点

- **高效**
	- 显著减少可训练参数（仅约 0.1%~1%），降低计算和存储成本。
- **快速训练**
	- LoRA 模块小，训练时间短，适合快速实验和原型设计。
- **模块化部署**
	- LoRA 参数可独立保存和加载，不影响原始模型结构。
- **兼容性强**
	- 可与量化（如 QLoRA）、混合精度等方法结合使用。
- **适用于大模型**
	- 在不修改原模型的基础上微调巨型模型（如LLaMA、GPT、Qwen等）。

### 4.2 缺点

- **适配能力有限**
	- 由于只调整部分模块，适应复杂任务时可能不如全参数微调。
- **层选择敏感**
	- 效果依赖于插入 LoRA 的层和位置，需经验或实验确定。
- **某些模型结构不适配**
	- 如卷积网络或结构不清晰的Transformer变体。
- **推理时需加载额外模块**
	- 部署需同时加载主模型与 LoRA 模块。
