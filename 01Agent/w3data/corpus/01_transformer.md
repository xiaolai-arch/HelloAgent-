# Transformer 架构

Transformer 是 2017 年论文《Attention Is All You Need》提出的架构，完全抛弃了循环结构，只靠注意力机制建模序列中任意两个位置之间的关系。

## 整体结构

原始 Transformer 是 Encoder-Decoder 结构：

- **Encoder（编码器）**：由 N=6 层相同的层堆叠而成，每层包含两个子层——多头自注意力（Multi-Head Self-Attention）和前馈网络（FFN），每个子层外面套残差连接（Residual Connection）和 LayerNorm。
- **Decoder（解码器）**：也是 N=6 层，多了一个"交叉注意力"子层，让解码端的每个位置都能看到编码端的全部输出。解码器使用掩码自注意力（Masked Attention），保证生成第 t 个词时看不到第 t 个词之后的内容。

## 数据流向

输入文本先经过分词器（Tokenizer）变成 token 序列，再查嵌入表变成向量，加上位置编码（Positional Encoding）后进入堆叠层。每一层的处理都是：注意力负责"横向"信息交换（token 之间互相看），FFN 负责"纵向"逐位置变换（每个 token 独立过同一个两层 MLP）。最后的输出向量接一个线性层投影到词表大小，再过 softmax 得到下一个 token 的概率分布。

## 为什么比 RNN 好

1. **并行**：RNN 必须按时间步串行计算，第 t 步依赖第 t-1 步；自注意力对所有位置一次性算完，训练时能吃满 GPU。
2. **长程依赖**：RNN 信息要一步一步传，距离远了会衰减；自注意力任意两个位置之间的路径长度是 O(1)，直接建立连接。
3. **可解释的计算图更简单**：堆相同的层即可扩规模，催生了后来的 GPT（Decoder-only）、BERT（Encoder-only）、T5（Encoder-Decoder）三大路线。

## 主流模型的取舍

现代大模型基本都是 Decoder-only 结构，只保留掩码自注意力 + FFN 的堆叠，用 causal mask 一次性完成训练。位置编码也从绝对正弦编码进化为 RoPE（旋转位置编码）等相对位置方案，以支持长上下文外推。
