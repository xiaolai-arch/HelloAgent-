# w2day03 · System Prompt vs User Prompt

## 本目录内容

| 文件 | 说明 |
|---|---|
| `anthropic_prompt_notes.md` | Anthropic Prompt 指南学习笔记：核心原则、System/User 分工、专业角色写法、自检清单 |
| `system_prompt_code_reviewer.md` | **主产出**：一套完整的“资深 Python 代码审查专家”System Prompt（角色定义 / 行为边界 / 输出规范） |
| `system_vs_user_demo.py` | 对比实验脚本：同一段代码，无 System vs 有 System |
| `demo_report.md` | 实验报告（含两组完整输出与结论） |
| `demo_result.json` | 原始输出、token 用量与自动评分数据 |

## 一句话结论

> **System 定“人设和规矩”，User 给“材料和任务”。**

专业角色 System Prompt 真正起作用的部分不是“你是专家”这句话，而是：

1. 角色定义：服务谁、擅长什么、不做什么
2. 行为边界：严重级别、缺信息怎么办、安全红线、敏感值处理
3. 输出规范：固定章节、固定字段、排序规则、长度上限

## 快速运行

```bash
python 01Agent/w2day03/system_vs_user_demo.py
```

脚本会读取 `../../vision_config.json`（或环境变量），调用 2 次模型，并重新生成 `demo_report.md` 与 `demo_result.json`。

## 本次实测摘要

| 组别 | 得分 | 结构 | 关键风险覆盖 | 泄露完整假密钥 |
|---|---|---|---|---|
| 无 System（普通提问） | 30.0 | 0/5 | 3/3 | 是 |
| 有 System（专业角色） | 100.0 | 5/5 | 3/3 | 否 |

无 System 组其实也识别出了 SQL 注入、硬编码密钥、密码哈希泄露等问题，但没有固定结构，并且把假密钥原样贴进了输出。有 System 组严格按五段式输出、使用 Blocker/Major/Minor 分级，并对敏感值做了掩码。

> 注意：代码里的密钥是测试用假值；System Prompt 不是安全边界，真正的密钥保护必须在代码层实现。

## 资料来源说明

`anthropic.com/learn`、`docs.anthropic.com`、`platform.claude.com/docs` 在当前网络环境返回区域限制（App unavailable in region）。笔记依据可访问的 Anthropic 官方工程博客、AWS Bedrock 官方 Prompt 指南（引用 Anthropic 文档）以及官方文档的公开搜索摘要整理。详见 `anthropic_prompt_notes.md` 第 0 节。
