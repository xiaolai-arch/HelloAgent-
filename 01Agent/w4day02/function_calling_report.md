# Function Calling 完整闭环实验报告（W4 Day2）

- 生成时间：2026-10-21 16:49:44
- 模型：`mimo-v2.6-flash`（temperature=0）
- 工具：`get_weather(city, unit)`（模拟数据源，确定值可复现）

## 一、FC 四步流程（与 Day1 手写 ReAct 对照）

```text
①定义工具        请求体 tools 字段挂 JSON Schema
      │
      ▼
②模型决策        返回 message.tool_calls（finish_reason=tool_calls）
      │          模型只产出"调用意图"，不执行任何函数
      ▼
③代码执行        你的代码解析 arguments → 查天气/算数/查库
      │
      ▼
④结果回传        追加 role=tool 消息（带 tool_call_id）→ 再请求
                 模型基于结果生成最终回答（finish_reason=stop）
```

| 维度 | Day1 手写 ReAct（文本协议） | Day2 Function Calling（API 协议） |
|---|---|---|
| 协议位置 | system prompt 里的格式约定 | 请求体 tools 字段的 JSON Schema |
| 模型输出 | 结构化文本（Thought/Action/...） | 结构化对象（tool_calls[].function） |
| 解析 | 正则 + 格式容错（Day1 出现格式违规） | json.loads arguments，字段由协议保证 |
| 参数校验 | 自己写 | Schema 声明类型/枚举/required |
| 决策自由度 | 相同：模型都可以选择不调用 | 相同（tool_choice=auto 时） |
| 本质 | 相同：模型给意图，代码干活，结果回上下文 ||

## 二、工具定义全文（进 API 协议层的那份 JSON）

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询指定城市当前天气（模拟数据源）",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "城市名，如 北京"
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "description": "温度单位，默认 celsius"
          }
        },
        "required": [
          "city"
        ]
      }
    }
  }
]
```

## 三、运行轨迹

### 幕 1：北京今天天气怎么样？适合穿什么？

- **hop1** finish_reason=`tool_calls`
  - 模型请求：`get_weather({"city": "北京"})`（call id: call_3389270d9da04736a3e95ada）
  - 代码执行：{'city': '北京'} → {'city': '北京', 'temperature': 9, 'unit': 'celsius', 'condition': '晴', 'wind': '西北风 3 级'}
- **hop2** finish_reason=`stop`
  - 最终回答：北京今天天气情况如下：

🌤 **天气**：晴  
🌡 **温度**：9°C  
💨 **风力**：西北风 3 级

**穿衣建议：**
- 早晚温差可能较大，建议穿**厚外套或夹克**，如风衣、薄羽绒服
- 内搭可以选择**毛衣或卫衣**
- 西北风 3 级有一定风感，怕冷的话可以加一条**围巾**
- 晴天紫外线较

### 幕 2：不需要工具的问题

- 问题：用一句话解释什么是注意力机制
- 结果：finish_reason=`stop`，tool_calls=无——工具列表在场，模型按 auto 语义自行决定不用。FC 不是"必须调工具"，而是"可以调工具"。

### 幕 3：多轮 FC（追问）

- **user**: 北京今天多少度？
  - finish_reason=`tool_calls`，calls=['{"city": "北京"}']
  - 回答：北京今天的天气情况如下：

- **温度**：9°C
- **天气**：晴
- **风力**：西北风 3 级

天气晴朗，但气温较低，注意保暖！
- **user**: 那上海呢？
  - finish_reason=`tool_calls`，calls=['{"city": "上海"}']
  - 回答：上海今天的天气情况如下：

- **温度**：17°C
- **天气**：多云
- **风力**：东风 2 级

比北京暖和不少，适合出行！

## 四、观察与结论

1. **finish_reason 是循环的开关**：幕 1 的轨迹是 ['tool_calls', 'stop']——
   tool_calls 表示"等我执行完再回来"，stop 表示"说完了"。
   Agent 主循环就是盯着这个字段决定继续还是退出。
2. **工具定义有 token 成本**：幕 1 第一跳 prompt=141（含 tools Schema），最终跳=230（历史又加上了 tool 结果）。
   工具越多 Schema 越肥，这正是 W2 上下文工程"预算"视角的具体案例。
3. **tool 消息的配对规则**：assistant 的 tool_calls 必须原样回填，
   执行结果用 role=tool + tool_call_id 逐条对应——错配会被网关拒绝，
   这是 FC 历史管理最常见的坑。
4. **多轮时模型在第 3 轮能再次发起调用**：历史里的 tool 消息不是一次性脚手架，而是对话的正式公民；模型可以基于前几轮的工具结果
   决定继续查新城市。

## 五、复现方式

```bash
python 01Agent/w4day02/function_calling.py
```
