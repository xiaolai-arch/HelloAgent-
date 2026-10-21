# Anthropic 官方 Prompt 指南学习笔记

> 主题：System Prompt vs User Prompt 的分工，以及如何写一个“专业角色”System Prompt。

## 0. 关于资料访问的说明

本次尝试直接访问了以下 Anthropic 官方入口：

- `https://www.anthropic.com/learn`
- `https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview`
- `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/system-prompts`

当前网络环境返回的是 **“App unavailable in region”**（区域限制），拿不到正文。因此笔记主要依据以下可访问的官方/半官方材料整理：

1. Anthropic 官方工程博客：**Effective context engineering for AI agents**（2025-09-29）  
   https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
2. AWS Bedrock 官方文档：**Prompt engineering concepts**（其中明确引用并链接 Anthropic Claude 的 Prompt 指南）  
   https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html
3. Anthropic 官方文档在搜索引擎中的公开摘要（prompt engineering overview、system prompts、XML tags、role prompting 等条目）。

这不是“凭印象编”的二手总结：核心原则（清晰直接、结构化分隔、示例、角色设定、right altitude、token 预算）都能在上述官方材料中对应到。等网络条件允许时，建议再对照 docs 原文补一遍。

---

## 1. Anthropic 指南的几条核心原则

### 1.1 清晰、直接、具体

不要写“帮我看看这段代码”，而要写清楚：

- 你是什么角色、服务谁
- 这次任务要产出什么
- 有哪些硬约束
- 输出用什么结构
- 什么情况算完成、什么情况算失败

Anthropic 反复强调：模型不会读心，**模糊的指令换来模糊的输出**。

### 1.2 用结构分隔不同语义

Anthropic 推荐用 XML 标签或 Markdown 标题把 Prompt 拆成清晰的区块，例如：

```xml
<role>...</role>
<instructions>...</instructions>
<context>...</context>
<examples>...</examples>
<output_format>...</output_format>
```

这样做的目的不是“好看”，而是降低模型混淆指令、上下文、示例和输入的概率。Claude 对 XML 标签尤其敏感，官方文档也专门推荐过 `<example>`、`<document>` 等标签。

### 1.3 示例要“少而典型”

Few-shot 示例是官方明确推荐的技术，但不要把所有 edge case 都塞进去。Anthropic 的建议是：挑选**多样、典型、能代表期望行为**的例子。例子是“给模型看的图片”，一张好图胜过一堆规则。

### 1.4 角色设定（Role Prompting）

在 System Prompt 里设定角色，可以稳定模型的语气、知识视角和判断标准。但“你是一位专家”本身不够，必须搭配：

- 具体职责
- 行为边界
- 输出规范
- 判断标准

否则角色只是一句空话。

### 1.5 Right Altitude（合适的高度）

Anthropic 用“right altitude”形容 System Prompt 的粒度：

- **太低**：把复杂业务逻辑写成 if-else 硬编码，Prompt 脆弱、难维护。
- **太高**：只给“你要专业、要准确”这种空话，模型没有可执行的信号。
- **合适**：具体到能指导行为，又保留模型根据上下文判断的空间。

官方原话大意：System Prompt 应该“specific enough to guide behavior effectively, yet flexible enough to provide the model with strong heuristics”。

### 1.6 上下文是有限预算

Anthropic 把 context 视为有限资源：token 越多，模型的注意力越容易被稀释（context rot）。所以 System Prompt 不是越长越好，而是要找**最小的高信号 token 集合**。能删的客套话、重复规则、无效例子都应删掉。

### 1.7 让模型“先想再做”（按需）

Chain-of-Thought 仍然有效，但不是所有任务都需要。对推理模型来说，内部已经有思维链，外部再强制“逐步思考”可能只是重复消耗 token。判断标准是：任务是否需要多步推导、是否存在明显陷阱。

### 1.8 System Prompt 不是安全边界

即使把“不要泄露密钥”写进 System Prompt，也不能把它当成真正的安全控制。Prompt injection、越权、密钥泄露要靠代码层、权限层、密钥管理来解决。System Prompt 只能降低概率，不能提供保证。

---

## 2. System Prompt vs User Prompt：分工

### 2.1 一张表看懂

| 维度 | System Prompt | User Prompt |
|---|---|---|
| **位置** | Anthropic Messages API 的顶层 `system` 参数；OpenAI 兼容接口里是 `{"role": "system"}` 消息 | `messages` 里的 `user` 消息 |
| **生命周期** | 整个会话/整个应用稳定生效 | 每一轮单独变化 |
| **主要作用** | 定义“你是谁、为谁服务、守什么规矩、怎么输出” | 提供“这一轮要处理什么” |
| **内容类型** | 角色、目标、边界、语气、输出 schema、安全策略、工具规则 | 具体问题、代码、文档、数据、临时参数 |
| **优先级** | 更高，模型被训练为优先遵循 | 较低，不能覆盖 System 的硬约束 |
| **变化频率** | 低（版本化、可复用） | 高（每轮都不同） |
| **调试入口** | 行为不对 → 改 System | 任务不对 → 改 User |
| **缓存友好** | 稳定前缀，利于 prompt caching | 变化部分，放在后面 |

### 2.2 一句话记忆

> **System 定“人设和规矩”，User 给“材料和任务”。**

### 2.3 什么放 System，什么放 User

**放 System：**

- 角色定义：你是谁、有多少经验、服务什么对象
- 长期目标：这类任务要达成什么
- 行为边界：不做什么、遇到不确定怎么办、拒绝什么
- 输出规范：章节、表格、字段、长度、语言
- 语气与风格：正式/简洁/教学式
- 安全与合规：隐私、密钥、越权、内容政策
- 工具使用规则（如果应用有工具）

**放 User：**

- 本轮要审查的代码、要总结的文档、要回答的问题
- 本轮的业务背景、数据规模、环境版本
- 本轮的临时参数：目标读者、字数、语气微调
- 本轮希望重点关注的模块

**不要放 System：**

- 每一轮都会变的任务数据（会破坏缓存，也让 Prompt 臃肿）
- 完整密钥、密码、token
- 大段原始日志/文档（放 User，必要时用 RAG）

**不要放 User：**

- 重复一遍 System 已经写死的角色和输出规范（除非本轮要临时覆盖，且 System 允许）
- 与任务无关的闲聊

### 2.4 Anthropic API 与 OpenAI 兼容接口的差异

- **Anthropic Messages API**：`system` 是请求体的顶层参数，不是一条 `role: system` 的消息。
- **OpenAI 兼容接口**（本仓库使用的网关）：System 是 `messages` 数组里的第一条 `{"role": "system", "content": ...}`。

两者只是协议形状不同，**概念上的分工是一样的**：System 放稳定行为，User 放本轮任务。

### 2.5 一个常见误区

> “System Prompt 是隐藏的，所以可以放密钥。”

错。System Prompt 不是保险箱：

- 它仍然进入模型上下文，可能被 prompt injection 套出来；
- 日志、调试工具、第三方网关都可能记录它；
- 真正的密钥应放在服务端环境变量/密钥管理系统，永远不要写进 Prompt。

---

## 3. “专业角色”System Prompt 的推荐结构

一个可复用的专业角色 System Prompt，至少包含六块：

```text
1. 角色定义（Role）
   - 你是谁、经验、服务对象
   - 你擅长什么、不擅长什么

2. 目标（Goal）
   - 这类任务要交付什么结果
   - 什么叫“做完了”

3. 行为边界（Boundaries）
   - 不做什么
   - 缺信息怎么办
   - 不确定怎么表达
   - 安全/合规红线
   - 严重级别/判断标准

4. 输出规范（Output Specification）
   - 固定章节/字段/表格
   - 排序规则、长度、语言
   - 必须包含什么、禁止包含什么

5. 工作流程（Workflow）
   - 先看什么、再看什么
   - 如何从输入走到输出
   - 自检步骤

6. 初始化（Initialization）
   - 收到输入后第一句话说什么
   - 不要自我介绍、不要寒暄
```

这套结构不是死的。任务简单时可以合并；任务复杂时可以加 `<examples>`、`<tools>`、`<reminders>`。但**角色、边界、输出**这三块最好不要省，因为它们是“专业感”和“可预测性”的主要来源。

---

## 4. 写 System Prompt 的常见错误

| 错误 | 后果 | 修法 |
|---|---|---|
| 只写“你是资深专家” | 输出和普通提问没区别 | 补职责、边界、输出规范 |
| 规则互相冲突 | 模型随机选一条执行 | 给规则排优先级，冲突时说明以谁为准 |
| 输出格式只靠形容词 | 每次结构都不一样 | 给固定模板/表格/字段清单 |
| 把所有 edge case 写成 if-else | Prompt 脆弱、难维护 | 用原则 + 少量典型示例 |
| 把任务数据塞进 System | 缓存失效、Prompt 臃肿 | 数据放 User |
| 不写“不确定怎么办” | 模型编造事实 | 明确要求标注“待确认” |
| 不写“不要做什么” | 模型过度发挥、越界 | 加行为边界和拒绝策略 |
| System 里放密钥 | 泄露风险 | 密钥只放服务端密钥管理 |
| 一次给太多目标 | 注意力分散、每项都做不好 | 一个 System 只服务一类任务 |

---

## 5. 自检清单

写完一个专业角色 System Prompt，逐条问自己：

- [ ] 角色是否具体到“能判断什么该做、什么不该做”？
- [ ] 是否写清了服务对象和使用场景？
- [ ] 是否有明确的行为边界（不做什么、缺信息怎么办）？
- [ ] 是否有可复制的输出结构（章节/表格/字段）？
- [ ] 是否规定了严重级别或判断标准？
- [ ] 是否规定了语言、长度、代码标识符的处理方式？
- [ ] 是否写了“不确定时怎么办”，避免编造？
- [ ] 是否有安全/隐私红线？
- [ ] 是否把本轮任务数据留在了 User Prompt？
- [ ] 是否删掉了客套话、重复规则和无效示例？
- [ ] 是否能用 3~5 个测试用例验证它真的遵守规范？

---

## 6. 本目录产出

| 文件 | 内容 |
|---|---|
| `system_prompt_code_reviewer.md` | 一套完整的“资深 Python 代码审查专家”System Prompt，包含角色定义、行为边界、输出规范、工作流程 |
| `system_vs_user_demo.py` | 对比实验：同一个代码审查任务，无 System vs 有 System |
| `demo_report.md` | 实验报告（运行后生成） |
| `demo_result.json` | 原始输出与指标（运行后生成） |

运行方式：

```bash
python 01Agent/w2day03/system_vs_user_demo.py
```

配置读取方式与前面几天一致：`vision_config.json` 优先，环境变量兜底。
