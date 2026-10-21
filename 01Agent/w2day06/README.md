# w2day06 · W2 收尾与 Context Engineering 延伸

本目录是 W2 的收尾产出，包含四块：

1. **3 类 Prompt 模板（共 5 个文件）**：整理成可直接复用的模板库，放到仓库。
2. **W2 学习笔记**：把 Day1~Day3 的实验数据、结论和知识点串成体系。
3. **W3 RAG 预习**：开课前的地图，包含两条流水线、关键决策、评估指标和 5 天动手计划。
4. **Context Engineering 面试话术**：从 Prompt Engineering 到 Context Engineering 的行业转变，重点讲预算、压缩、记忆分层，并给出可背诵的话术。

---

## 一、文件导航

| 文件 | 内容 | 适合什么时候看 |
|---|---|---|
| [`prompts/README.md`](prompts/README.md) | 3 套模板的总览、选择指南、变量、使用方式 | 想直接拿模板用 |
| [`prompts/01-langgpt-role-coach.md`](prompts/01-langgpt-role-coach.md) | 角色型：LangGPT 概念教练 | 教学、概念解释 |
| [`prompts/02-zero-shot-task.md`](prompts/02-zero-shot-task.md) | 推理型：Zero-shot | 简单任务、最省 token |
| [`prompts/02-few-shot-task.md`](prompts/02-few-shot-task.md) | 推理型：Few-shot | 锁格式、锁语气 |
| [`prompts/02-cot-reasoning.md`](prompts/02-cot-reasoning.md) | 推理型：Chain-of-Thought | 多步推理、需审计 |
| [`prompts/03-system-code-reviewer.md`](prompts/03-system-code-reviewer.md) | 专业型：代码审查 System Prompt | 生产级代码审查 |
| [`prompts/render_prompt.py`](prompts/render_prompt.py) | 模板渲染脚本（标准库） | 填变量、导出 Prompt |
| [`W2学习笔记.md`](W2学习笔记.md) | W2 知识体系 + 实验复盘 + 自测题 | 复习、面试前 |
| [`W3_RAG预习.md`](W3_RAG预习.md) | RAG 全景、评估、5 天计划 | W3 开课前 |
| [`Context_Engineering_面试话术.md`](Context_Engineering_面试话术.md) | 行业转变、5R 框架、十问十答、W16 路线 | 面试准备、阶段四 |

---

## 二、任务完成清单

- [x] 整理 3 套 Prompt 模板，放到 GitHub 仓库
- [x] 整理 W2 笔记（Day1~Day3 实验 + 知识点补齐）
- [x] 预览 W3 的 RAG 材料（公开资料整理，非课程原文）
- [x] 延伸：Prompt Engineering → Context Engineering（压缩 / 预算 / 记忆分层）
- [x] 面试用新话术表述老知识（30 秒 / 60 秒 / 2 分钟 / STAR / 十问十答）
- [x] 阶段四 W16 展开清单

---

## 三、一页速览

### W2 做了什么

| 实验 | 结论 | 代价 |
|---|---|---|
| Day1 LangGPT 结构化 Prompt | 结构命中 5/5 vs 0~1/5 | prompt tokens 约 30x；推理预算可能被挤占 |
| Day2 Zero/Few/CoT | CoT 100 分；Few-shot 最便宜；Zero-shot 是默认 | CoT 825 tokens vs Few-shot 450 |
| Day3 System vs User | 有 System 100 分 vs 无 System 30 分 | 总 tokens 约 2.2x |

### Context Engineering 的三句话

1. **预算**：token、注意力、成本/延迟，三者一起算。
2. **压缩**：优先可恢复压缩（路径/URL/ID），摘要是有损兜底。
3. **记忆分层**：工作上下文 / 会话状态 / 长期语义·情景·程序性记忆 / 外部知识 / 模型参数。

### W3 准备

- 能说清 RAG 的离线索引和在线查询两条流水线。
- 能区分 dense / sparse / hybrid / rerank。
- 能定义 Recall@k、Faithfulness、Context Precision/Recall。
- 知道什么时候**不需要** RAG（小、静态、可整体放入上下文的语料）。

---

## 四、建议阅读顺序

**如果只有 15 分钟：**

1. 本 README 的“一页速览”
2. `W2学习笔记.md` 的“一页速览”和“八条可复用原则”
3. `Context_Engineering_面试话术.md` 的“先背这 5 句”和“30 秒版”

**如果有 1 小时：**

1. `W2学习笔记.md` 全文
2. `prompts/README.md` 的“怎么选”和“怎么验证”
3. `Context_Engineering_面试话术.md` 的“预算 / 压缩 / 记忆分层”
4. `W3_RAG预习.md` 的“两条流水线”和“评估”

**如果准备面试：**

1. `Context_Engineering_面试话术.md` 的“十问十答”
2. “STAR 案例”用自己的话复述一遍
3. “白板题”手画一遍信息流
4. 用 `W2学习笔记.md` 的数据支撑你的结论

---

## 五、怎么用模板

```bash
# 列出模板和变量
python 01Agent/w2day06/prompts/render_prompt.py --list

# 渲染概念教练 System Prompt
python 01Agent/w2day06/prompts/render_prompt.py 01-langgpt-role-coach.md \
  -v ROLE_NAME="RAG 概念教练" \
  -v DOMAIN="RAG 与检索增强生成" \
  -v AUDIENCE="有 Python 基础的后端工程师" \
  -v LANGUAGE="中文" \
  -v DEFAULT_LENGTH="300~450 字"

# 渲染 CoT Prompt
python 01Agent/w2day06/prompts/render_prompt.py 02-cot-reasoning.md \
  -v TASK="..." \
  -v CONTEXT="无外部上下文"
```

> Windows 控制台如果是 GBK，中文可能显示乱码；建议加 `-o prompt.txt` 输出到文件，或先 `chcp 65001` / 设置 `PYTHONUTF8=1`。

---

## 六、说明与边界

- **W2 实验数据**来自仓库中已有的 Day1~Day3 报告，模型为 `mimo-v2.6-flash`；样本较小，结论是趋势，不是普遍规律。
- **W3 预习**基于公开资料整理，仓库里没有 W3 官方讲义；课程材料下发后应以课程为准。
- **Context Engineering** 引用了 Anthropic、LangChain、Manus、Chroma 等公开资料；价格、比例、benchmark 数字会随版本变化，面试时讲量级和 trade-off，不要把别人的数据说成自己的生产数据。
- **Prompt 模板**是通用起点，不是银弹；上线前请用自己的测试集做 A/B，并记录正确率、结构、token、延迟。
- **安全**：Prompt 不是安全边界；密钥、权限、注入防护必须在代码层、权限层和工具层实现。

---

## 七、下一步

- W3：按 `W3_RAG预习.md` 的 5 天计划，做一个最小 RAG + 评估报告。
- 阶段四 W16：从 `Context_Engineering_面试话术.md` 的 8 个方向里选 2~3 个做深。
- 模板库：每次修改后跑一遍自己的测试集，记录“改了什么、为什么改、指标怎么变”。
