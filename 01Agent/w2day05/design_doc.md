# 活动报名系统技术方案 v0.1

- 作者：业务研发组
- 日期：2026-10-21
- 状态：待评审
- 上线窗口：2 周后

## 1. 背景与目标

公司每月举办 1~2 场线下活动，每场预计 3000~5000 人报名。目前用问卷收集名单，人工导入 Excel，效率低，容易出现重复报名和名额统计错误。

本期目标：2 周内上线一个在线报名系统，支持用户报名、名额控制、管理后台查看和导出报名列表。

本期非目标：支付、选座、候补、退款、发票。这些后续再做。

## 2. 流量与容量假设

- 单场活动 5000 个名额，开放报名后预计 10 分钟内报满。
- 峰值约 200 QPS，持续 10 分钟；平时接近 0。
- 报名表单 12 个字段，其中手机号、身份证号为必填。
- 管理后台同时在线约 20 人，导出操作用得不多。
- 短信服务平均响应 300ms，未设置超时和重试。

## 3. 技术选型

- 前端：Vue 3 + Vite
- 后端：Python 3.11 + FastAPI
- 数据库：SQLite，单文件放在服务器本地盘
- 部署：单台 2 核 4G 云服务器，Nginx + uvicorn 4 workers
- 定时任务：APScheduler，跟 Web 服务放在同一个进程
- 短信：调用第三方短信 API
- 缓存/队列：本期不引入，先保持简单

## 4. 数据模型

```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  start_time TEXT NOT NULL,        -- 例如 "2026-11-01 09:00:00"
  capacity INTEGER NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE signups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  phone TEXT NOT NULL,
  id_card TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

- 手机号、身份证号明文存储，方便客服查询和抽奖。
- 没有为 `signups.event_id` 建索引。
- 报名成功以 `signups` 插入成功为准。

## 5. 核心流程

1. 用户提交报名表单。
2. 后端查询活动当前报名人数：

   ```sql
   SELECT COUNT(*) FROM signups WHERE event_id = ?;
   ```

3. 如果人数小于 `capacity`，执行 `INSERT` 插入报名记录。
4. 同步调用短信 API 发送确认短信，等待返回后给用户响应。
5. 管理后台列表接口 `GET /api/events/{id}/signups` 返回该活动全部报名记录，前端做分页。

## 6. 接口设计

### POST /api/signups

请求体：`name`、`phone`、`id_card`、`event_id`

响应：

```json
{"code": 0, "message": "报名成功"}
```

### GET /api/events/{id}/signups

响应：该活动全部报名记录，包含姓名、手机号、身份证号。

## 7. 安全与运维

- 管理后台使用固定口令 `admin123`，写在前端环境变量里。
- 日志用 `logger.info("request=%s", request.body)` 打印完整请求体，方便排查。
- 未接入验证码、限流、风控。
- 数据库每天凌晨 3 点用 `cp` 备份到同一块磁盘。
- 没有监控和告警；出问题看日志。
- 上线方式：本地打包，`scp` 到服务器，重启 uvicorn。

## 8. 上线计划

- 第 1 周：开发接口和页面。
- 第 2 周：联调、压测（如果来得及）、上线。
- 回滚方案：保留上一版代码，出问题改回去。

## 9. 已知风险（作者自评）

- 短信服务可能偶尔慢，但一般能恢复。
- SQLite 够用，等活动量大了再换 PostgreSQL。
- 2 周时间紧，先保证功能可用。
