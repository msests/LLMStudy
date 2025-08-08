## 1. 概述

BLEU（Bilingual Evaluation Understudy）是机器翻译和文本生成任务中最早被广泛使用的自动评价指标之一，由 IBM 研究院在 2002 年提出。它的核心思想是通过比较机器生成的翻译和一组人工参考翻译之间的 _n-gram_ 重合度，来衡量生成文本的质量。

主要特点：
- **自动化**：不需要人工参与即可得分。
- **语言无关**：适用于多种语言对。
- **高效**：计算简单，可用于大规模评估。

## 2. 原理

BLEU 评分的核心基于以下几个概念：

### 2.1 n-gram 精确度（Precision）

- 比较生成文本（hypothesis）中的 n-gram 与参考文本（reference）中的 n-gram 的匹配数。
- 对每种 n-gram，最多只能匹配参考中出现的次数，防止重复“刷分”。
$$Precision_n=\frac{\sum_{ngram \in H} \min(\text{count}_{H}(ngram), \text{count}_{R}(ngram))}{\sum_{ngram \in H} \text{count}_{H}(ngram)}$$

### 2.2 加权几何平均

通常将 1-gram 到 4-gram 的精确度进行加权几何平均：
$$P= \exp\left(\sum_{n=1}^N w_n \log \text{Precision}_n\right)$$
其中通常 $w_n= \frac{1}{N}$

### 2.3 惩罚项：Brevity Penalty（简短惩罚）

避免模型通过只输出很短的翻译来获得高分。定义如下：
$$BP=\begin{cases} 1 & \text{if } c > r \\ e^{(1 - \frac{r}{c})} & \text{if } c \le r \end{cases}$$
其中 $c$ 是候选句长度，$r$ 是参考句最接近的长度。

### 2.4 BLEU 分数

最终 BLEU 分数是：
$$BLEU = BP \cdot P$$
## 3. 实现

可以使用如下方式在 Python 中实现 BLEU：

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'test']

score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                      smoothing_function=SmoothingFunction().method1)
print(f"BLEU score: {score:.4f}")
```

**常用库**
- `nltk.translate.bleu_score`
- `sacrebleu`（标准化 BLEU，推荐用于论文和实验）

## 4. 优缺点

### 4.1 优点

- **计算快速**
	- 适合大规模自动评估。
- **标准化程度高**
	- 可用于模型比较。
- **语言无关**
	- 适用于任意语言对。

### 4.2 缺点

- **忽略语义**
	- BLEU 只比较表面 n-gram，不理解语义。
- **对句子长度敏感**
	- 需要惩罚项来避免短句偏高评分。
- **不适合句子级评价**
	- BLEU 本质是为语料级设计，单句结果不稳定。
- **无法处理同义词**
	- 不同表达方式得分可能很低。
