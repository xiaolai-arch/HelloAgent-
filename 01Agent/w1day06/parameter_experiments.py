# -*- coding: utf-8 -*-
"""采样参数实验：temperature / max_tokens / top_p 的真实效果

对应 w1day03 笔记《tokens_采样_实验指南.md》第 2 节的理论，这次全部用真 API 验证：
- temperature：改分布形状（尖 ↔ 平），不截断候选 —— 理论上 T=0 复跑应一致，
  实测托管网关不保证确定性（见实验 1 的实测注记）
- max_tokens：硬上限，撞上就截断 —— finish_reason 会从 "stop" 变成 "length"；
  且本模型是思考型模型，思考+回答共享这份预算（见实验 2）
- top_p：按累计概率截断候选集（核采样）—— p 越小越保守，p 越大越发散

实验设计思路（控制变量法，和 w1day03 的对比实验一脉相承）：
  实验 1  固定提示词，只变 temperature，每档跑两次看随机性
  实验 2  固定提示词与温度，只变 max_tokens（三档递增），观察截断与思考预算
  实验 3  固定 temperature=1.0，只变 top_p，隔离出核采样的独立效果

执行顺序（各家实现略有差异）：先 temperature 调 logits → 再 top-p 截断 → 最后抽样。
所以业界建议：两个采样参数通常只调一个，另一个保持默认。

配置同 first_llm_call.py：../../vision_config.json 或环境变量。
"""
import sys
import io
import os
import json
import urllib.request
import urllib.error

# Windows 终端默认 GBK，强制 UTF-8 输出防中文乱码（与 w1day03/w1day05 一致）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---- 配置加载（同 first_llm_call.py，密钥只放 gitignore 忽略的配置文件） ----
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json")

_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
MODEL = _cfg.get("model", "mimo-v2.6-flash")

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH} 或设置环境变量。")
print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")


def chat(messages, temperature=0.7, top_p=1.0, max_tokens=512):
    """带全部采样参数的 OpenAI 兼容调用，返回 (resp, error_msg) 元组。

    与 first_llm_call.py 的 chat() 相比多了 top_p，且不直接退出 ——
    返回错误信息而不是 sys.exit，让实验循环能在个别调用失败时继续跑完。

    三个采样参数在请求体里的位置（全部可选，不传即用厂商默认）：
      temperature —— w1day03：softmax(logits / T)，T 越小分布越尖
      top_p       —— 累计概率截断：候选 = 概率从高到低累加首次 ≥ p 的集合
      max_tokens  —— 生成（思考+回答）总量的硬上限
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")  # dict → JSON str → UTF-8 字节流
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,  # 带 data 参数 → 自动 POST
        headers={"Authorization": "Bearer " + API_KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r), None  # 成功：(响应 dict, None)
    except urllib.error.HTTPError as e:
        # 失败：(None, 错误描述)。常见 400=该网关不接受某参数组合
        detail = e.read().decode("utf-8", errors="replace")[:300]
        return None, f"HTTP {e.code}: {detail}"


def ask(prompt, **kw):
    """便捷函数：单条 user 消息发一次请求。

    返回 (可见文本, finish_reason, usage dict)。
    只取 message.content —— 思考型模型还有 reasoning_content（内部思维链），
    实验观察的是用户可见输出，不取思维链。
    """
    resp, err = chat([{"role": "user", "content": prompt}], **kw)
    if err:
        return f"<<调用失败: {err}>>", "-", {}
    c = resp["choices"][0]
    return c["message"]["content"], c.get("finish_reason", "?"), resp.get("usage", {})


print("=" * 62)
print("实验 1：temperature —— 同一提示，同一温度，跑两次")
print("=" * 62)
# 提示词选择有讲究：要挑"有多个合理答案"的任务（起名字）。
# 温度作用于每个位置的概率分布 —— 如果问事实题（如"中国的首都是?"），
# 分布本来就极尖，什么温度答案都一样，实验现象出不来。
# 每档温度跑两次：一次看不出随机性，两次的差异才是温度的直接效果。
P = "给我的宠物机器人起一个名字，只说名字，不说别的。"
for t in [0.0, 0.7, 1.5]:
    # 三档覆盖理论三态：T→0（贪心/近确定性）、T=1（原始分布）、T>1（拉平/发散）
    a1, _, _ = ask(P, temperature=t)
    a2, _, _ = ask(P, temperature=t)
    same = "→ 两次一致" if a1.strip() == a2.strip() else "→ 两次不同"
    print(f"\n[temperature={t}]")
    print(f"  第1次: {a1.strip()[:60]}")
    print(f"  第2次: {a2.strip()[:60]}")
    print(f"  {same}")
# 实测结论（重要，修正教科书式的预期）：本网关 T=0 两次输出也不同。
# 原因：托管推理是服务化的，batch 组合、浮点归约顺序、思考型模型内部采样
# 都会引入非确定性；要严格复现需厂商提供 seed/确定性模式。
# 但相对趋势不变：T 越小 → 分布越尖 → 输出越倾向高概率的保守词。
print("\n[note] 实测（mimo-v2.6-flash 网关）：即使 T=0，两次输出也不同。")
print("       T=0 的语义是'每次取当前分布的最高概率词'，但托管推理服务通常")
print("       不保证确定性：batch 组合、浮点归约顺序、思考型模型的内部采样")
print("       都会让同一请求跑出不同结果。要严格复现需厂商的 seed/确定性模式。")
print("       相对趋势仍然成立：T 越小分布越尖，越倾向保守高概率词。")

print()
print("=" * 62)
print("实验 2：max_tokens —— 硬上限与截断（finish_reason=length）")
print("=" * 62)
# "一直往下编故事"制造长输出需求，让上限轻松被撞到。
# 三档递增（20 → 200 → 1000）：20 必截断；200 观察"够不够"；1000 看完整产出。
LONG = "请从「从前有座山」开始，一直往下编故事，越长越好。"
for cap in [20, 200, 1000]:
    text, reason, usage = ask(LONG, temperature=0.7, max_tokens=cap)
    got = usage.get("completion_tokens", "?")
    # 思考型模型的消耗明细：completion_tokens 里包含 reasoning_tokens
    think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    print(f"\n[max_tokens={cap}]  finish_reason={reason!r}  生成={got} tokens"
          f"  其中思考={think}")
    if text.strip():
        print(f"  开头: {text[:80].strip()}...")
        # finish_reason == "length" 是判断"被截断"的可靠信号，
        # 比猜文本结尾像不像句子更工程化 —— 生产代码应检查这个字段。
        if reason == "length":
            print(f"  结尾: ...{text[-50:].strip()}")
            print("  → 句子被拦腰截断 —— max_tokens 是生成侧的刹车，不管语法完整。")
    else:
        # 本模型是思考型模型：先产出 reasoning_content（思维链）再产出 content。
        # 预算全被思考吃掉时，可见回答就是空字符串 —— 这是思考型模型特有的坑。
        print(f"  可见回答为空！{think} 个 token 全部花在了内部思考上，")
        print("  还没开始正式回答就撞了上限 —— 思考型模型的 max_tokens 是")
        print("  思考+回答共享的总预算。")
print("\n[note] 本模型（mimo-v2.6-flash）是思考型模型：message 里除了 content 还有")
print("       reasoning_content（思维链），usage.completion_tokens_details.reasoning_tokens")
print("       记录思考消耗。max_tokens 不影响 prompt，只限制'思考+输出'的总生成量。")
print("       且思考长度每次不同（同样 200 上限，有时够用有时全被思考吃掉），")
print("       给思考型模型设 max_tokens 要留足余量。")
print("       输出上限通常 < context window（w1day03：200K 上下文的模型输出可能只有 32K~64K）。")

print()
print("=" * 62)
print("实验 3：top_p（核采样）—— 固定 temperature=1.0，只动 top_p")
print("=" * 62)
# 控制变量：temperature 固定 1.0（即"不调温度"），这样观察到的差异
# 全部来自 top_p 的候选截断，而不是两个参数混在一起的效果。
# w1day03 的比喻回顾：Top-p = 按概率从高到低排队，站进队伍累计到 p 的人入选。
P3 = "用一个比喻解释神经网络，只说一句话。"
for p in [0.1, 0.9, 1.0]:
    # 每档跑两次，对比输出的多样性
    outs = [ask(P3, temperature=1.0, top_p=p)[0].strip()[:70] for _ in range(2)]
    same = "两次一致" if outs[0] == outs[1] else "两次不同"
    print(f"\n[top_p={p}]  ({same})")
    print(f"  第1次: {outs[0]}")
    print(f"  第2次: {outs[1]}")
# 解读时注意：p=0.1 截断的是"每个位置"的候选集，不是"整句话只准一种答案"。
# 每个 token 位置仍从剩下的少数候选里抽样，加上思考型模型内部还有一层采样，
# 所以句子级别仍会多样 —— 截断变小 = 每步更保守，不等于全局唯一解。
print("\n[note] w1day03 理论回顾：Top-p 按概率从高到低累加，首次 ≥ p 的集合就是候选。")
print("       p=0.1 → 可能只剩 1~2 个词；p=1.0 → 全词表都保留（只受 temperature 影响）。")
print("       注意：p 小 ≠ 全局确定 —— 截断发生在每个 token 位置，整句仍可多样。")

print()
print("=" * 62)
print("速查表：参数怎么选")
print("=" * 62)
print("""
  任务类型          temperature    top_p        max_tokens
  ---------------  -------------  -----------  ------------------
  抽取/分类/代码    0 ~ 0.2        1（不动）     按输出长度上限设
  通用对话          0.7 ~ 1.0      0.9（默认档） 留足余量
  创意写作          1.0 ~ 1.3      0.95         尽量大

  通用原则：temperature 和 top_p 通常只调一个；
            max_tokens 与 finish_reason 配合判断「是否被截断」；
            思考型模型的 max_tokens 是 思考+回答 的总预算，要额外留余量。
""")
