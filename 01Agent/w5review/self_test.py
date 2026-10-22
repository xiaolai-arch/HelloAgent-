# -*- coding: utf-8 -*-
"""阶段一自测脚本（复习周）——口述演练前的知识体检

对应任务："口述练习：ReAct / RAG / MCP / Transformer；录音回听找逻辑不清的地方"
用法：
  python self_test.py            # 交互模式：逐题自答 → 回车看参考答案 → 自评
  python self_test.py --list     # 只列出全部题目
  python self_test.py --w1       # 只抽 W1 的题（可选 w1/w2/w3/w4）

题库覆盖阶段一四周：LLM 原理（W1）/ Prompt 与上下文工程（W2）/
RAG（W3）/ Agent·FC·MCP（W4）。参考答案里凡是本仓库实验得出的数据，
都标注了出处脚本——口述时能用自己跑出来的数字举例，含金量完全不同。
"""
import sys
import io
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BANK = [
    # ---- W1 LLM 原理 ----
    {"w": "w1", "q": "白板画出 Transformer 的整体结构，标注数据流向。",
     "a": "输入 → Tokenizer → Embedding(+位置编码) → N×[掩码自注意力 → FFN，各带残差+LayerNorm] → 线性层 → softmax → 下一个 token 概率。现代 LLM 是 Decoder-only，用 causal mask 保证训练并行。参考：01Agent/w1day01。"},
    {"w": "w1", "q": "自注意力的 Q/K/V 各是什么角色？为什么要除以 sqrt(d_k)？",
     "a": "Q=我在找什么，K=我能提供什么的索引，V=实际内容。除以 sqrt(d_k) 是因为维度越大点积方差越大，不缩放会把 softmax 推入饱和区、梯度消失。参考：01Agent/w1day02、w3day01 手算例子。"},
    {"w": "w1", "q": "温度调高为什么输出更随机？Top-p 和 Top-K 的区别？",
     "a": "温度给 logits 除以 T 再 softmax：T 越大分布越平，低概率 token 机会变大。Top-p 按累计概率动态截断（窗口大小随分布变），Top-K 固定截前 K 个。参考：01Agent/w1day03。"},
    {"w": "w1", "q": "MoE 里'总参数量'和'激活参数量'分别决定什么？",
     "a": "总参数量决定容量上限，激活参数量决定推理成本/速度。DeepSeek-V3：671B 总 / 37B 激活。代价：负载均衡难、显存占用高（所有专家要驻留）。参考：01Agent/w1day04、w3 语料 04。"},
    {"w": "w1", "q": "多模态'视觉编码器 + LLM'架构的三个部件是什么？LLaVA 用了什么？",
     "a": "视觉编码器（ViT）→ 投影层（对齐到语言空间）→ LLM 统一自回归生成。LLaVA = CLIP ViT + 线性投影 + Vicuna/LLaMA。接口上只是 content 里多了 image_url part。参考：01Agent/w1day05。"},

    # ---- W2 Prompt / 上下文工程 ----
    {"w": "w2", "q": "LangGPT 结构化提示词的思路是什么？代价是什么？",
     "a": "用类编程结构（Role/Profile/Skills/Constraints/Workflow/OutputFormat）写角色说明书，输出结构命中率显著高于随意写；代价是 prompt tokens 成倍增加（实测约 30 倍）。参考：01Agent/w2day01。"},
    {"w": "w2", "q": "Zero-shot / Few-shot / CoT 各适合什么场景？",
     "a": "Zero-shot 默认起手最省 token；Few-shot 锁格式锁语气（示例进上下文）；CoT 用于多步推理和需审计的过程，代价是输出 token 多、延迟高。参考：01Agent/w2day02。"},
    {"w": "w2", "q": "System Prompt 和 User Prompt 怎么分工？",
     "a": "System 是整段对话的'宪法'（角色/边界/输出规范），放数组最前，优先级高于用户消息；User 是每轮具体输入。实测有 System 的任务完成率显著更高。参考：01Agent/w2day03。"},
    {"w": "w2", "q": "JSON Mode 和 Function Calling 的区别一句话？",
     "a": "JSON Mode 约束输出形态（保证是合法 JSON），FC 约束输出意图（函数名+参数 JSON，Schema 进协议层）。JSON Mode 保证'是 JSON'，Schema 才保证'是哪个 JSON'。参考：01Agent/w2day04。"},
    {"w": "w2", "q": "Context Engineering 的三个核心问题是什么？（面试新话术）",
     "a": "预算（窗口/注意力/成本一起算）、压缩（优先可恢复压缩，摘要有损兜底）、记忆分层（工作上下文/会话状态/长期记忆/外部知识/模型参数）。参考：01Agent/w2day06。"},

    # ---- W3 RAG ----
    {"w": "w3", "q": "画出 RAG 的两条流水线。",
     "a": "离线索引：文档→解析→清洗→切分→元数据→向量化→向量索引；在线查询：问题→改写→召回(Dense+Sparse)→融合→重排→打包→生成→引用。离线做不好，在线调参没用。参考：01Agent/w3day03。"},
    {"w": "w3", "q": "余弦相似度怎么算？为什么文本检索都用它而不是欧氏距离？",
     "a": "cos(a,b)=(a·b)/(|a||b|)，衡量方向一致。bge 等模型输出已 L2 归一化，余弦=点积。它不受向量长度影响，只看语义方向。参考：01Agent/w3day01（含手算 0.707 的例子）。"},
    {"w": "w3", "q": "chunk 太小/太大各有什么问题？overlap 是干嘛的？",
     "a": "太小：单块信息太薄，凑不齐相关性（实测 recursive-150 的 Recall@5 只有 0.667）；太大：多主题混在一起向量被稀释、挤预算。overlap 让跨边界信息在两块各有一份完整拷贝，但会稀释 MRR（实测 overlap=50 比 0 的 MRR 低）。参考：01Agent/w3day04。"},
    {"w": "w3", "q": "BM25 和向量检索各自强在哪？为什么要 hybrid？RRF 好在哪？",
     "a": "BM25 强在精确词（错误码 TRV-40907、型号），向量强在同义改写（'退货'≈'退款'）。强项几乎不重叠所以 hybrid；RRF 只用排名不用分数（两路量纲不同没法加），score=Σ1/(k+rank)，k 常取 60。参考：01Agent/w3day05。"},
    {"w": "w3", "q": "Rerank 为什么比向量检索准？代价是什么？",
     "a": "cross-encoder 让 (query, chunk) 在模型内部互相看见（bi-encoder 是各自编码），精度高但每对都要跑一次模型，只能现场算没法预建索引。实测 MRR 0.639→0.717，代价是每题多约 3s（20 候选）。参考：01Agent/w3day05。"},
    {"w": "w3", "q": "RAGAS 四指标分别评什么？排查顺序？",
     "a": "Context Recall（该有的资料齐吗）→ Context Precision（相关的排前面吗）→ Faithfulness（答案被资料支持吗）→ Answer Relevancy（答了问题吗）。先检索后生成，锅甩对位置。参考：01Agent/w3day06。"},
    {"w": "w3", "q": "什么时候不需要 RAG？",
     "a": "知识库小且静态（整体放进上下文+缓存更好）、缺的是推理不是知识、数据结构化查询明确（SQL/API）、延迟极度敏感。先问：缺的是知识、推理、权限还是实时性？参考：01Agent/w3 语料 07。"},

    # ---- W4 Agent / FC / MCP ----
    {"w": "w4", "q": "口述 ReAct 完整流程，并说明为什么比纯 Reasoning 强。",
     "a": "Thought(缺什么)→Action(调工具)→Observation(真实事实回填)→循环→Final Answer。强在把不确定的部分外包给真实工具，推理从'凭记忆'变'凭证据'。实测：设 MAX_STEPS 防打转，文本协议会偶发格式违规。参考：01Agent/w4day01。"},
    {"w": "w4", "q": "Function Calling 四步流程？finish_reason 怎么用？",
     "a": "定义工具(tools+Schema)→模型决策(返回 tool_calls, finish_reason=tool_calls)→代码执行→role=tool 回传(带 tool_call_id)→模型生成最终回答(stop)。Agent 主循环盯着 finish_reason 决定继续还是退出。参考：01Agent/w4day02。"},
    {"w": "w4", "q": "并行工具调用的收益和前提是什么？",
     "a": "收益：墙钟时间从 sum(各工具) 变 max(各工具)；前提：调用之间无依赖。要点：模型一条消息给多个 tool_calls，客户端并发执行后一次性回传全部 role=tool 消息。写操作慎并行。参考：01Agent/w4day03。"},
    {"w": "w4", "q": "MCP 三层角色是什么？和 FC 什么关系？",
     "a": "Host(宿主应用，安全边界)、Client(与 Server 1:1 连接的中间层)、Server(暴露工具/资源/模板的轻量进程)。MCP 不替代 FC：Server 列出的工具最终仍以 FC 形式被模型决策调用，MCP 站在 FC 上游解决'工具写一次处处用'。参考：01Agent/w4day04、w4day05 实测 9/9。"},
    {"w": "w4", "q": "四大 Agent 模式怎么选？",
     "a": "从 ReAct 起步；路径可预先拆解→Plan-and-Execute；质量可迭代打磨→Reflection（设 max_iterations）；子任务真独立+单上下文装不下→才上 Multi-Agent。复杂度本身是成本。参考：01Agent/w4day06。"},
]


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for i, t in enumerate(BANK, 1):
            print(f"{i:>2}. [{t['w']}] {t['q']}")
        return

    weeks = [a for a in args if a.startswith("--w")]
    pool = [t for t in BANK if not weeks or any(t["w"] == w[2:] for w in weeks)]
    if not pool:
        print("没有匹配的题目（--w1/--w2/--w3/--w4）")
        return
    random.shuffle(pool)

    print(f"阶段一口述自测：共 {len(pool)} 题（回车看参考答案，自评 y=答上了 / n=没答上）")
    print("=" * 62)
    good, bad, skipped = [], [], []
    for i, t in enumerate(pool, 1):
        print(f"\n[{i}/{len(pool)}] · {t['w'].upper()} · {t['q']}")
        try:
            input("  （自己先口述，然后回车看参考答案…）")
        except (EOFError, KeyboardInterrupt):
            print("\n[中断] 本轮成绩不保存")
            return
        print(f"  参考答案：{t['a']}")
        try:
            verdict = input("  答上了吗？(y/n/s=跳过) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[中断] 本轮成绩不保存")
            return
        if verdict == "y":
            good.append(t["q"])
        elif verdict == "s":
            skipped.append(t["q"])
        else:
            bad.append(t["q"])

    print()
    print("=" * 62)
    print(f"小结：答上 {len(good)} / 没答上 {len(bad)} / 跳过 {len(skipped)}")
    if bad:
        print("\n没答上的题（明天再测一遍这几道）：")
        for q in bad:
            print(f"  - {q}")
    if good:
        print("\n已经能口述的题：录音复述一遍，回听找逻辑不清的地方。")
    print("\n验收标准（阶段一）：能流利口述 ReAct 流程 / RAG 流程 / Transformer 架构。")


if __name__ == "__main__":
    main()
