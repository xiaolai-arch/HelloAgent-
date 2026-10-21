# Prompt 模板库（W2 产出）

> 3 类可直接复用的 Prompt 模板（共 5 个文件），覆盖「角色型 / 推理型 / 专业型」三类高频场景。
> 所有模板用 `{{VAR}}` 表示变量，配套 `render_prompt.py` 可一键渲染。

## 一、3 套模板是什么

| 套件 | 文件 | 解决什么问题 | W2 实测依据 |
|---|---|---|---|
| **01 角色型**：LangGPT 概念教练 | `01-langgpt-role-coach.md` | 把抽象概念讲清楚，稳定输出「定义 → 类比 → 机制 → 误区 → 自测」 | Day1：结构化组结构命中 5/5，普通组 0~1/5 |
| **02 推理型**：Zero-shot / Few-shot / CoT | `02-zero-shot-task.md`、`02-few-shot-task.md`、`02-cot-reasoning.md` | 根据任务难度、格式要求和 token 预算选择推理策略 | Day2：CoT 100 分 / 825 tokens；Few-shot 450 tokens；Zero-shot 593 tokens |
| **03 专业型**：代码审查 System Prompt | `03-system-code-reviewer.md` | 用 System Prompt 固定角色、边界、严重级别和输出规范 | Day3：有 System 100 分 vs 无 System 30 分；总 tokens 2.2x |

三套模板不是三个孤立的“提示词”，而是三种**上下文组织方式**：

1. **角色型**：用稳定的角色、目标和输出格式，降低输出的随机性。
2. **推理型**：用示例或步骤，控制模型的推理路径和 token 花费。
3. **专业型**：用 System Prompt 把“人设、规矩、输出契约”从每轮任务中解耦出来。

## 二、怎么选

```text
任务需要稳定的角色/输出规范？
├─ 是 → 03 专业型 System Prompt
└─ 否
   ├─ 目标是“让人学会一个概念”？ → 01 角色型概念教练
   └─ 目标是“把任务做对/做完”？
      ├─ 任务简单、模型裸跑就够 → 02 Zero-shot
      ├─ 输出格式/语气/字段必须稳定 → 02 Few-shot
      └─ 多步推导、容易算错/漏条件 → 02 CoT
```

补充规则：

- **示例比规则更擅长锁格式，但不擅长教推理。** 如果示例只给“问题 → 答案”，模型学到的是输出形状，不是推导过程。
- **CoT 不是越多越好。** 推理模型（reasoning model）内部已经有思维链，外部再强制“逐步思考”可能只是重复消耗 token。
- **Zero-shot 不是“差”。** 简单任务上它最便宜、最快；复杂任务才需要示例或推理引导。
- **一个 System Prompt 只服务一类任务。** 不要试图用一个“万能专家”覆盖代码审查、写作、数据分析。

## 三、变量约定

模板中的变量统一写成 `{{UPPER_SNAKE_CASE}}`。渲染时未提供且未加 `--allow-missing` 的变量会报错，避免把 `{{TASK}}` 原样发给模型。

### 01 角色型

| 变量 | 含义 | 示例 |
|---|---|---|
| `{{ROLE_NAME}}` | 角色名 | `RAG 概念教练` |
| `{{DOMAIN}}` | 角色所属领域 | `RAG 与检索增强生成` |
| `{{AUDIENCE}}` | 目标读者 | `有 Python 基础、刚接触 RAG 的后端工程师` |
| `{{LANGUAGE}}` | 输出语言 | `中文` |
| `{{DEFAULT_LENGTH}}` | 默认篇幅 | `300~450 字` |

配套 User Prompt（不要和 System Prompt 混在一起发）：

```text
请解释这个概念：{{TOPIC}}
我的背景：{{AUDIENCE}}
我希望重点了解：{{FOCUS}}
```

### 02 推理型

| 变量 | 含义 | 说明 |
|---|---|---|
| `{{TASK}}` | 本轮任务 | 题目、指令、要处理的输入；三个推理模板都使用 |
| `{{CONTEXT}}` | 本轮材料 | 文档、代码、数据；没有就写“无外部上下文” |
| `{{OUTPUT_FORMAT}}` | 输出契约 | Zero-shot / Few-shot 必填；CoT 模板已内置固定格式 |
| `{{EXAMPLES}}` | Few-shot 示例 | 仅 `02-few-shot-task.md` 使用 |

Few-shot 示例块的推荐写法：

```text
示例 1
输入：...
输出：...

示例 2
输入：...
输出：...
```

示例要**少而典型**：2~5 个，覆盖主要模式，不要把所有 edge case 都塞进去。示例太多会让模型模仿示例的表面模式，而不是理解规则。

### 03 专业型

`03-system-code-reviewer.md` 是一个完整的、可直接使用的 System Prompt，没有变量。需要改的通常只有：

- 角色年限和技术栈（默认 Python 后端 / 12 年）
- 严重级别的定义
- 输出章节和问题数量上限
- 团队特有的安全红线

配套 User Prompt：

```text
请审查下面这段代码，找出会在生产环境造成问题的地方。

<context>
- 语言/框架版本：{{VERSION}}
- 部署环境：{{ENVIRONMENT}}
- 数据规模/QPS：{{SCALE}}
- 希望重点关注：{{FOCUS}}
</context>

<code>
{{CODE}}
</code>
```

> 上面的 `{{VERSION}}`、`{{ENVIRONMENT}}`、`{{SCALE}}`、`{{FOCUS}}`、`{{CODE}}` 属于 User Prompt，不在 System 模板里，`render_prompt.py --list` 不会列出它们；发送前请用代码或手工替换。

## 四、快速使用

### 4.1 列出模板和变量

```bash
python 01Agent/w2day06/prompts/render_prompt.py --list
```

### 4.2 渲染模板

```bash
# 01 角色型
python 01Agent/w2day06/prompts/render_prompt.py 01-langgpt-role-coach.md \
  -v ROLE_NAME="RAG 概念教练" \
  -v DOMAIN="RAG 与检索增强生成" \
  -v AUDIENCE="有 Python 基础、刚接触 RAG 的后端工程师" \
  -v LANGUAGE="中文" \
  -v DEFAULT_LENGTH="300~450 字"

# 02 CoT
python 01Agent/w2day06/prompts/render_prompt.py 02-cot-reasoning.md \
  -v TASK="某班有 50 名学生，30 人参加数学竞赛……问只参加一门的有多少人？" \
  -v CONTEXT="无外部上下文"

# 渲染到文件
python 01Agent/w2day06/prompts/render_prompt.py 02-zero-shot-task.md \
  -v TASK="总结这段代码的风险" \
  -v CONTEXT="..." \
  -v OUTPUT_FORMAT="## 结论\n## 关键依据\n## 最终答案" \
  -o prompt.txt
```

> Windows PowerShell 下如果引号转义麻烦，建议把长文本先写进文件，再用 `--var TASK="$(Get-Content task.txt -Raw)"`，或者直接用 Python 调用 `render()`。

### 4.3 OpenAI 兼容接口

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

system = open(
    "01Agent/w2day06/prompts/01-langgpt-role-coach.md",
    encoding="utf-8",
).read()

user = """请解释这个概念：RAG
我的背景：有 Python 基础、刚接触 RAG 的后端工程师
我希望重点了解：检索、重排、评估"""

resp = client.chat.completions.create(
    model=os.environ.get("MODEL", "your-model"),
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    temperature=0.3,
    max_tokens=2000,
)
print(resp.choices[0].message.content)
```

### 4.4 Anthropic Messages API

Anthropic 的 `system` 是请求体的顶层参数，不是一条 `role: system` 的消息：

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

msg = client.messages.create(
    model=os.environ["ANTHROPIC_MODEL"],
    system=system,
    messages=[{"role": "user", "content": user}],
    max_tokens=2000,
)
print(msg.content[0].text)
```

概念上的分工是一样的：**System 放稳定行为，User 放本轮任务。**

## 五、怎么验证模板真的有用

不要凭“看起来不错”判断。做一个最小对照实验：

1. **固定变量**：同一模型、同一温度、同一 `max_tokens`、同一组测试输入。
2. **只改一个东西**：有无 System、Zero/Few/CoT、模板版本 A/B。
3. **准备 5~10 个测试用例**：至少包含 1 个边界情况、1 个对抗输入、1 个“信息不足”的情况。
4. **分开打分**：
   - 正确性（答案对不对）
   - 结构（章节/字段是否齐全）
   - 约束遵守（长度、语言、敏感值处理）
   - 成本（prompt / completion / reasoning tokens）
5. **记录失败模式**：是没理解任务、没遵守格式、编造事实，还是上下文不够。

W2 的三组实验就是按这个思路做的：

- [Day1 LangGPT 对比](../../w2day01/langgpt_compare_report.md)
- [Day2 Zero-shot / Few-shot / CoT 对比](../../w2day02/prompt_strategies_report.md)
- [Day3 System vs User 对比](../../w2day03/demo_report.md)

## 六、设计原则（从 W2 实验里提炼）

1. **先写输出契约，再写形容词。** “专业、详细、结构化”不如直接给章节和字段。
2. **角色不是一句“你是专家”。** 真正起作用的是职责、边界、判断标准、输出规范。
3. **示例要少而典型。** 示例是“给模型看的图片”，不是“把规则再写一遍”。
4. **稳定前缀放 System，变化负载放 User。** 这既符合职责分工，也有利于 prompt caching。
5. **控制“合适的高度”（right altitude）。** 太低是把 if-else 写进 Prompt，太高是只给空话。
6. **上下文是有限预算。** Prompt 不是越长越好，要找最小的高信号 token 集合。
7. **Prompt 不是安全边界。** 密钥、权限、注入防护必须在代码层和权限层解决。
8. **用数据说话。** 正确率、结构命中率、token 成本、延迟，至少记录两个。

## 七、常见错误

| 错误 | 后果 | 修法 |
|---|---|---|
| 只写“你是资深专家” | 输出和普通提问没区别 | 补职责、边界、输出规范 |
| 规则互相冲突 | 模型随机选一条执行 | 给规则排优先级，冲突时说明以谁为准 |
| 输出格式只靠形容词 | 每次结构都不一样 | 给固定模板/表格/字段清单 |
| 把所有 edge case 写成 if-else | Prompt 脆弱、难维护 | 用原则 + 少量典型示例 |
| 把任务数据塞进 System | 缓存失效、Prompt 臃肿 | 数据放 User |
| 不写“不确定怎么办” | 模型编造事实 | 明确要求标注“待确认” |
| 一次给太多目标 | 注意力分散、每项都做不好 | 一个 System 只服务一类任务 |
| 把密钥写进 Prompt | 泄露风险 | 密钥只放服务端密钥管理 |
| 示例全是同一种模式 | 模型被 few-shot 进“沟”里 | 增加多样性，覆盖边界 |

## 八、来源与致谢

- LangGPT 的结构（Role / Profile / Goals / Skills / Rules / Workflow / OutputFormat / Initialization）受 [LangGPT](https://github.com/langgptai/LangGPT) 项目启发，本项目中的模板为适配 W2 学习场景做了改写。
- Prompt 设计原则参考 Anthropic 的 Prompt Engineering 文档与工程博客：
  - [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  - [Prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- 代码审查模板的安全边界、严重级别和输出规范，来自 W2 Day3 的实验与复盘。

## 九、版本

- `v1.0`：W2 Day1~Day3 的三份实验产物。
- `v1.1`：抽出为 3 套模板，补充变量、渲染脚本、选择指南和验证方法。

模板不是一成不变的。每次修改后，至少跑一遍自己的测试集，并记录“改了什么、为什么改、指标怎么变”。
