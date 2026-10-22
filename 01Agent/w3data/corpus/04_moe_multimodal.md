# MoE 与多模态：主流模型的两个进化方向

## MoE：稀疏混合专家

MoE（Mixture of Experts，混合专家）把 Transformer 的每个 FFN 层替换成多个"专家"FFN 加一个路由器（Router）：每个 token 只被送给得分最高的少数几个专家（如 8 选 2），其余专家不参与计算。

关键指标要区分两个数：

- **总参数量**：所有专家加起来的规模，决定模型"容量"上限。
- **激活参数量**：每个 token 实际用到的参数，决定推理成本和速度。

DeepSeek-V3 总参数 671B、每 token 只激活 37B——容量接近千亿级大模型，推理成本却是几十亿级。代价是训练要小心负载均衡（防止所有 token 都涌向少数专家，其他专家荒废），显存占用高（所有专家都要驻留显存）。

## 多模态：视觉编码器 + LLM

图文多模态模型的主流架构是"视觉编码器 + 投影层 + LLM"：

1. 图像先过视觉编码器（如 ViT，Vision Transformer）变成一串视觉特征向量；
2. 投影层（MLP 或 Cross-Attention）把视觉特征对齐到语言模型的嵌入空间；
3. LLM 把视觉向量和文本 token 一起当输入，统一做自回归生成。

代表模型：GPT-4o（原生多模态端到端）、Qwen-VL / Qwen2.5-VL（Qwen 视觉系列）、LLaVA（开源经典方案：LLaVA = 视觉编码器 CLIP ViT + 线性投影 + Vicuna/LLaMA）。接口形态上，多模态只是把消息 content 从字符串换成数组，多了一个 image_url 类型的 part，chat/completions 协议形状完全不变。

## 横向对比记忆锚点

- GPT-4o：OpenAI 旗舰，原生多模态，综合能力强，闭源。
- DeepSeek-V3：MoE 路线代表，开源权重，性价比高。
- Qwen2.5：阿里全家桶（文本/视觉/代码/数学），开源生态最全。
- LLaMA：Meta 开源系鼻祖，社区微调生态最丰富。
