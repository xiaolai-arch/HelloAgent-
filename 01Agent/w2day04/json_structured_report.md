# JSON 结构化输出机制对比实验报告（JSON Mode vs Function Calling）

- 生成时间：2026-10-21 15:23:56
- 模型：`mimo-v2.6-flash`
- 采样参数：temperature=0.2, max_tokens=2000
- 文本数：3 × 方法数：4 = 12 次调用
- 任务：抽取 人名 / 日期 / 事件 + 一句话摘要

## 一、JSON Mode 和 Function Calling 的区别

### 1.1 一句话区分

- **JSON Mode**：约束**输出的形态**——保证 `content` 是一段合法 JSON。
  至于字段叫什么、类型对不对、内容准不准，API 层不管。
- **Function Calling**：约束**输出的意图**——模型以“我要调用某个函数”的形式回答，
  产出 函数名 + 参数 JSON，参数结构由挂在 `tools` 上的 JSON Schema 描述。
  模型并不真的执行函数，执行方是你的代码。

### 1.2 格式对比

| 维度 | JSON Mode | Function Calling |
|---|---|---|
| 开启方式 | `response_format: {"type": "json_object"}` | `tools: [...]` + `tool_choice` |
| Schema 写在哪 | API 层不带 Schema，只能写在 Prompt 里（严格模式除外） | `tools[0].function.parameters`，直接传 JSON Schema |
| 模型返回位置 | `choices[0].message.content` | `choices[0].message.tool_calls[0].function.arguments` |
| 返回的内容 | 一段 JSON 文本 | 函数名 + 参数 JSON 文本 |
| finish_reason | `stop` | `tool_calls` |
| 保证什么 | 语法合法（是 JSON） | 参数按 Schema 生成、函数名来自候选列表 |
| 不保证什么 | 字段齐全、类型正确、内容准确 | 参数值真实准确（模型可能幻觉） |
| 对话能否继续 | 通常就此结束 | 可把执行结果以 `role=tool` 回传继续对话 → 多轮工具循环 |
| 典型场景 | 信息抽取、格式转换、结构化写作 | Agent 调工具：搜索、下单、查库、执行代码 |
| 额外成本 | 几乎为零 | tools 定义计入 prompt tokens |

### 1.3 请求 / 响应长什么样

两种机制的请求体只有一小块不同（本实验脚本里全部收敛在 `build_extra()`）：

```json
// JSON Mode：加一个字段
{
  "model": "...",
  "messages": [{"role": "user", "content": "……"}],
  "response_format": {"type": "json_object"}
}

// Function Calling：把 Schema 挂到 tools 上，并强制调用
{
  "model": "...",
  "messages": [{"role": "user", "content": "……"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "save_extraction",
      "description": "保存从文本中抽取的结构化信息",
      "parameters": { ...同一段 JSON Schema... }
    }
  }],
  "tool_choice": {"type": "function", "function": {"name": "save_extraction"}}
}
```

响应里的 JSON 藏的位置不同：

```json
// JSON Mode 响应：JSON 字符串藏在 content 里，finish_reason=stop
{
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "{\"people\": [\"李明\"], \"dates\": [...], ...}"
    }
  }]
}

// Function Calling 响应：JSON 字符串藏在 arguments 里，finish_reason=tool_calls
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "save_extraction",
          "arguments": "{\"people\": [\"李明\"], \"dates\": [...], ...}"
        }
      }]
    }
  }]
}
```

关键观察：**两者最终都产出一段 JSON 字符串**，区别只在——

1. 位置不同：`content` vs `tool_calls[0].function.arguments`；
2. 语义不同：一段数据 vs 一次函数调用；
3. Schema 的位置不同：Mode 不带 Schema（只能写进 Prompt），FC 的 Schema 是 API 请求体的一部分。

### 1.4 两者的关系与演进

- Function Calling 的对话循环：模型给 `arguments` → 你的代码执行真正的函数 → 把结果以 `role: "tool"` 消息发回 → 模型基于结果继续生成。JSON Mode 没有这个循环，
  它只是一次性输出。
- 可以把 FC 理解为 JSON Mode 的“意图版”：都产出 JSON，但 FC 把“要什么字段、什么类型”
  升级成了 API 协议的一部分（`parameters` 就是 JSON Schema）。
- 新一代 API 把两者合体：OpenAI Structured Outputs（`response_format: json_schema`）
  和 Anthropic 的 strict tool use，都是“Schema 进协议 + 受限解码”，保证 100% 语法合规。

## 二、实验设计

- **控制变量**：同一个模型、同 3 段文本、temperature=0.2、max_tokens=2000；
  plain / json_mode / function_calling 三种方法用**同一条基础指令**，只改 API 层参数。
- **自变量**：结构化输出的约束机制。Schema 本身对四种方法完全相同。

| 方法 | Prompt | API 层参数 | 想验证什么 |
|---|---|---|---|
| plain | 只说“以 JSON 格式输出” | 无 | 最裸的口头约定有多不可靠 |
| json_mode | 同 plain | `response_format=json_object` | Mode 保证可解析，但保证 Schema 吗 |
| schema_prompt | 第 2 套模板（Schema 写进 Prompt） | 无 | Prompt 层给 Schema 能做到什么程度 |
| function_calling | 同 plain | `tools` + 强制 `tool_choice` | 协议层给 Schema 字段是不是最稳 |

评分规则（100 分）：

- JSON 可解析 40 分：裸输出直接 `json.loads` 成功 40；需要修复（去围栏/截取子串）25；失败 0
- Schema 校验 30 分：required 齐全、类型正确、无多余字段，每个错误扣 10
- 内容准确 30 分：人名/日期/事件三类 F1 平均 × 30；人名日期按“双向包含”匹配，
  事件按备选关键词命中；日期必须保留原文写法（写成 2026-09-15 会判不匹配）

### 测试文本与期望值

- **T1（科技发布会新闻）**：2026年9月15日，星海科技在上海举办了秋季发布会，正式推出新一代智能助手“小……
  - 人名：李明、王芳｜日期：2026年9月15日、2026年12月1日、2025年3月｜事件关键词：秋季发布会/发布会；API接口/API；启动研发/研发
- **T2（体育赛事新闻）**：2026年6月20日晚，亚洲青年田径锦标赛在日本大阪开幕。中国选手陈昊在男子11……
  - 人名：陈昊、佐藤健、刘国栋｜日期：2026年6月20日、2027年8月、2026年7月10日｜事件关键词：田径锦标赛开幕/锦标赛/开幕；13秒18夺冠/13秒18；返回北京集训/集训
- **T3（历史事件段落（含干扰信息））**：1919年5月4日，北京爆发五四运动，学生领袖傅斯年担任游行总指挥。1921年7……
  - 人名：傅斯年、王尽美、毛泽东｜日期：1919年5月4日、1921年7月、1935年1月｜事件关键词：五四运动/五四；一大/全国代表大会/中共一；遵义会议

## 三、Prompt 模板与 JSON Schema 全文

### 3.1 基础指令（plain / json_mode / function_calling 共用）

```text
请从下面的文本中抽取所有的人名、日期和事件，并给出一句话摘要，以 JSON 格式输出。

文本：
{{TEXT}}
```

### 3.2 第 2 套结构化 Prompt 模板（Schema 约束，schema_prompt 方法使用）

第 1 套是 w2day01 的 LangGPT 结构化模板（管“人设与回答风格”）；
这一套管“机器可校验的输出格式”：把 JSON Schema 直接写进 Prompt，
并用硬性约束条款钉死解析细节。`{{SCHEMA}}`、`{{TEXT}}` 是占位符，脚本运行时替换，
换一个 Schema 就能复用到任何抽取任务：

````markdown
# 角色
你是一个严谨的信息抽取引擎，只输出 JSON，除此之外不输出任何文字。

# 任务
从文本中抽取：人名、日期、事件，并给出一句话摘要。

# 输出格式（必须严格符合以下 JSON Schema）
```json
{{SCHEMA}}
```

# 硬性约束
1. 输出必须是一个合法 JSON 对象，可被 json.loads 直接解析；
2. required 字段全部必须出现，不得新增 Schema 之外的字段；
3. 人名、日期必须使用文本中的原文，禁止改写、翻译或换格式（例如“2026年9月15日”不得写成“2026-09-15”）；
4. 事件用简短短语概括，短语中必须保留文本里的关键词；
5. 某类信息在文本中不存在时返回空数组，禁止编造；
6. 直接输出 JSON 本体，不要用 Markdown 代码块包裹，不要加任何解释。

# 待抽取文本
{{TEXT}}
````

### 3.3 目标 JSON Schema（四种方法完全相同）

```json
{
  "type": "object",
  "properties": {
    "people": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "文本中出现的人名，保留原文"
    },
    "dates": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "文本中出现的日期/时间，保留原文写法"
    },
    "events": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "文本中发生的事件，简短短语"
    },
    "summary": {
      "type": "string",
      "description": "一句话摘要"
    }
  },
  "required": [
    "people",
    "dates",
    "events",
    "summary"
  ],
  "additionalProperties": false
}
```

## 四、实验结果

### 4.1 汇总（按方法）

| 方法 | 成功 | 裸解析 | 需修复 | 解析失败 | Schema通过 | 人名F1 | 日期F1 | 事件F1 | 平均分 | 平均prompt | 平均completion | tool_calls占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 纯 Prompt 约定 | 3/3 | 0 | 2 | 1 | 0/3 | 0.00 | 0.00 | 0.00 | 23.3 | 123 | 352 | 0% |
| JSON Mode | 3/3 | 3 | 0 | 0 | 0/3 | 0.00 | 0.00 | 0.00 | 40.0 | 123 | 254 | 0% |
| Schema 约束 Prompt | 3/3 | 3 | 0 | 0 | 2/3 | 0.67 | 0.67 | 0.45 | 78.0 | 515 | 182 | 0% |
| Function Calling | 3/3 | 3 | 0 | 0 | 3/3 | 1.00 | 1.00 | 0.86 | 98.7 | 323 | 197 | 100% |

### 4.2 逐条结果

| 文本 | 方法 | JSON来自 | 解析 | 解析分 | Schema | 内容分 | 总分 | completion | 思考 | finish_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 纯 Prompt 约定 | content | failed | 0 | ❌(20) | 0 | **20** | 462 | 32 | stop |
| T1 | JSON Mode | content | raw | 40 | ❌(0) | 0 | **40** | 249 | 32 | stop |
| T1 | Schema 约束 Prompt | content | raw | 40 | ❌(0) | 0 | **40** | 115 | 6 | stop |
| T1 | Function Calling | tool_calls | raw | 40 | ✅ | 30 | **100** | 226 | 23 | tool_calls |
| T2 | 纯 Prompt 约定 | content | fenced | 25 | ❌(0) | 0 | **25** | 287 | 30 | stop |
| T2 | JSON Mode | content | raw | 40 | ❌(0) | 0 | **40** | 214 | 32 | stop |
| T2 | Schema 约束 Prompt | content | raw | 40 | ✅ | 25 | **95** | 234 | 10 | stop |
| T2 | Function Calling | tool_calls | raw | 40 | ✅ | 26 | **96** | 197 | 23 | tool_calls |
| T3 | 纯 Prompt 约定 | content | fenced | 25 | ❌(0) | 0 | **25** | 308 | 161 | stop |
| T3 | JSON Mode | content | raw | 40 | ❌(0) | 0 | **40** | 299 | 186 | stop |
| T3 | Schema 约束 Prompt | content | raw | 40 | ✅ | 29 | **99** | 197 | 80 | stop |
| T3 | Function Calling | tool_calls | raw | 40 | ✅ | 30 | **100** | 168 | 13 | tool_calls |

### 4.3 同一段文本（T1）四种方法的原始输出对比

**纯 Prompt 约定**（content，stop）

```json
```json
{
  "人物": [
    {
      "姓名": "李明",
      "身份": "星海科技创始人"
    },
    {
      "姓名": "王芳",
      "身份": "星海科技首席科学家"
    }
  ],
  "日期": [
    {
      "日期": "2026年9月15日",
      "相关事件": "星海科技在上海举办秋季发布会，正式推出新一代智能助手"小星""
    },
    {
      "日期": "2026年12月1日",
      "相关事件": "小星正式开放API接口"
    },
    {
      "日期": "2025年3月",
      "相关事件": "小星项目启动研发"
    }
  ],
  "事件": [
    {
      "事件描述": "星海科技在上海举办秋季发布会，推出新一代智能助手"小星"",
      "涉及人物": ["李明", "王芳"],
      "涉及日期": ["2026年9月15日"]
    },
    {
      "事
……（截断，全文见 result.json）
```

**JSON Mode**（content，stop）

```json
{
  "人名": [
    "李明",
    "王芳"
  ],
  "日期": [
    "2026年9月15日",
    "2026年12月1日",
    "2025年3月"
  ],
  "事件": [
    "星海科技在上海举办秋季发布会，正式推出新一代智能助手'小星'",
    "创始人李明宣布小星将于2026年12月1日正式开放API接口",
    "首席科学家王芳介绍项目从2025年3月启动研发，历时18个月",
    "研发期间团队走访了200多家企业客户"
  ],
  "摘要": "2026年9月15日，星海科技在上海秋季发布会上由创始人李明和首席科学家王芳共同推出新一代智能助手'小星'，该项目历时18个月研发，计划于2026年12月1日正式开放API接口。"
}
```

**Schema 约束 Prompt**（content，stop）

```json
{"type":"object","properties":{"people":{"type":"array","items":{"type":"string"},"description":"文本中出现的人名，保留原文"},"dates":{"type":"array","items":{"type":"string"},"description":"文本中出现的日期/时间，保留原文写法"},"events":{"type":"array","items":{"type":"string"},"description":"文本中发生的事件，简短短语"},"summary":{"type":"string","description":"一句话摘要"}},"required":["people","dates","events","summary"],"additionalProperties":false}
```

**Function Calling**（tool_calls，tool_calls）

```json
{"people": ["李明", "王芳"], "dates": ["2026年9月15日", "2026年12月1日", "2025年3月"], "events": ["星海科技在上海举办秋季发布会并推出新一代智能助手“小星”", "李明宣布小星将于2026年12月1日开放API接口", "王芳介绍项目自2025年3月启动研发，历时18个月，走访200多家企业客户"], "summary": "2026年9月15日，星海科技在秋季发布会上推出智能助手“小星”，创始人李明宣布其API将于2026年12月1日开放，首席科学家王芳介绍了历时18个月的研发历程。"}
```

### 4.4 Schema 校验错误明细

- T1 · 纯 Prompt 约定：JSON 解析失败，无法校验
- T1 · JSON Mode：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 人名；$: Schema 之外的多余字段 日期；$: Schema 之外的多余字段 事件；$: Schema 之外的多余字段 摘要
- T1 · Schema 约束 Prompt：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 type；$: Schema 之外的多余字段 properties；$: Schema 之外的多余字段 required；$: Schema 之外的多余字段 add…
- T2 · 纯 Prompt 约定：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 人名；$: Schema 之外的多余字段 日期；$: Schema 之外的多余字段 事件；$: Schema 之外的多余字段 一句话摘要
- T2 · JSON Mode：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 人名；$: Schema 之外的多余字段 日期；$: Schema 之外的多余字段 事件；$: Schema 之外的多余字段 摘要
- T3 · 纯 Prompt 约定：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 人名；$: Schema 之外的多余字段 日期；$: Schema 之外的多余字段 事件；$: Schema 之外的多余字段 一句话摘要
- T3 · JSON Mode：$: 缺少 required 字段 people；$: 缺少 required 字段 dates；$: 缺少 required 字段 events；$: 缺少 required 字段 summary；$: Schema 之外的多余字段 人名；$: Schema 之外的多余字段 日期；$: Schema 之外的多余字段 事件；$: Schema 之外的多余字段 摘要

## 五、结论与讨论

- **平均分最高**：Function Calling（98.7/100）。
- **纯 Prompt 约定 2/3 次需要修复**：口头要求“输出 JSON”挡不住模型加 ``` 围栏或加说明文字，这就是 JSON Mode 存在的理由。
- **纯 Prompt 约定的字段名完全失控**：3 次输出的顶层键全是 人名/日期/事件/一句话摘要；T1 事件为扁平字符串，T2 事件为扁平字符串，T3 事件为扁平字符串。F1 全为 0 不是因为内容抽错——内容其实都在——而是字段名和结构每次都不一样，下游代码没有稳定契约可写。Schema 的价值正在这里：**不提高模型智商，只钉死契约。**
- **JSON Mode 全部裸解析通过（3/3 raw）**：它兑现了“语法合法”的承诺。
  但3 次输出的顶层键全是 人名/日期/事件/摘要——内容本身没错，却和 Schema 声明的 people/dates/events/summary 对不上，下游代码拿到手照样崩。原因很直白：指令是中文的，又没有任何机制钉死字段名，模型就按中文写键了。
  **这就是 JSON Mode 和 Schema 的分工：Mode 保证“是 JSON”，Schema 才保证“是哪个 JSON”。**
- **Schema 约束 Prompt（第 2 套模板）**：Schema 通过 2/3，平均分 78.0。它不依赖网关支持任何特殊参数，可移植性最好，代价是每条请求多花 Schema 文本的 prompt tokens（平均 prompt 515）。
- **Function Calling**：3/3 成功，JSON 来自 tool_calls 的比例 100%，Schema 通过 3/3。Schema 进了协议层，但参数值仍可能偏离原文（幻觉），内容分不会自动满分。

### 怎么选

1. **只要数据，不要动作**（抽取、翻译、格式转换）：JSON Mode 或 Schema 约束 Prompt 就够了；生产上再配 jsonschema/pydantic 校验 + 失败重试，因为 Mode 管语法不管内容。
2. **要动作**（Agent 调搜索、下单、执行代码）：必须 Function Calling——
   它的 `role=tool` 回传循环是 Agent 执行闭环的基础。可以说：
   **Function Calling 是 Agent 的手，JSON Mode 是 Agent 的嘴。**
3. **网关不支持 API 层机制时**：第 2 套 Schema 约束模板是纯 Prompt 的平替，任何 OpenAI 兼容端点都能用。
4. **三者可以叠加**：JSON Mode + Prompt 里写 Schema + 客户端校验重试，是很多生产管线的实际形态；structured outputs（json_schema 严格模式）则是把这条路做进 API 的终态。

### 实验局限

- 每种方法只有 3 段文本、各跑 1 次，样本小；temperature=0.2 下仍可能有波动。
- 判分里的 F1 用“双向包含/关键词命中”近似匹配，比精确匹配宽松，但比人工评判严格，规则全部公开可复现。
- 事件没有唯一正确答案，期望值是我们预设的关键词组，模型换一种概括方式会被扣分。
- 若网关对 json_object / tools 的支持程度与原生 OpenAI 不同，结论不能直接推广。

## 六、复现方式

```bash
python 01Agent/w2day04/json_structured_output.py
```

配置读取顺序：`vision_config.json` 优先，环境变量兜底；密钥不会写进脚本或报告。
