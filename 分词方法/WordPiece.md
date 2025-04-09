## 概述

WordPiece 是一种广泛应用于预训练语言模型（如BERT）的子词分词算法。其核心思想是**将单词拆分为高频出现的子词单元**，以平衡词表规模与未登录词（OOV）问题。与BPE（Byte Pair Encoding）的主要区别在于合并策略：WordPiece通过最大化语言模型概率选择合并对，而BPE基于相邻符号频率。

## 技术细节

### 算法流程

1. **初始化词汇表**  
   从基础字符（如ASCII字符+Unicode片段）开始，构建初始词表

2. **迭代合并子词**  
   - 对训练语料进行分词，统计所有可能的子词对
   - 计算每对子词$(A,B)$的合并得分：  
$$
score(A,B) = \frac{count(A,B)}{count(A) \times count(B)}
$$
   - **选择得分最高的子词对**进行合并，将新子词加入词表
   - 重复直到达到预设词表大小

3. **特殊标记处理**  
   - `[UNK]`：未识别字符
   - `[CLS]`/`[SEP]`：句子边界标记
   - `##`前缀：表示非起始子词

### 分词过程

采用**最长匹配优先策略**：
```python
def wordpiece_tokenize(word, vocab):
    tokens = []
    while len(word) > 0:
        # 从最长可能的前缀开始匹配
        max_len = min(len(word), max_subword_length)
        for i in range(max_len, 0, -1):
            subword = word[:i]
            if i == 1:  # 单字符处理
                subword = subword if subword in vocab else '[UNK]'
            else:
                subword = '##' + subword[1:] if i != len(word) else subword
                
            if subword in vocab:
                tokens.append(subword)
                word = word[i:]
                break
        else:
            tokens.append('[UNK]')
            break
    return tokens
```

### 数学模型

给定训练语料DD，优化目标为最大化似然函数：  
L=∏w∈DP(w)=∏w∈D∏s∈S(w)P(s)L=∏w∈D​P(w)=∏w∈D​∏s∈S(w)​P(s)  
其中S(w)S(w)表示单词ww的子词分割序列。通过EM算法迭代优化：

1. E步：固定词表，找最优分词路径
    
2. M步：固定分词结果，更新词表（合并最佳子词对）
    

## 3. 优点与缺点

### 优点

|特性|说明|
|---|---|
|OOV处理|通过子词组合可表达未见词汇|
|词表压缩|典型词表大小30k-50k（BERT: 30,522）|
|形态学保留|能捕捉词根、前后缀等结构特征|
|多语言支持|不依赖空格分词，适合黏着语言|

### 缺点

|问题|示例|影响|
|---|---|---|
|分词歧义|"play"可单独成词或拆分为"p+lay"|语义理解困难|
|重建困难|`['##ing', '##s']` → "ings"?|需要后处理规则|
|计算开销|最长匹配需要多次查表|比char-level慢3-5倍|
|标点敏感|"don't"可能拆分为["don","'","t"]|影响语义连贯性|

## 4. 典型应用示例

输入单词：`"unhappily"`  
分词结果：`["un", "##happi", "##ly"]`

处理逻辑：

1. 匹配最长有效子词`"un"`（在词表中）
    
2. 剩余部分`"happily"`继续匹配：
    
    - `"happily"`不在词表
        
    - `"happly"`（假设不存在）→ 最终拆分为`"##happi"+"##ly"`