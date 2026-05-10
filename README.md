# LifeEvolutionSimulation

基于 AI 模拟生物在外星球极端环境上演化的研究平台。

## 项目简介

科学严谨的外星球生态系统演化模拟器。配置星球参数（重力、大气、温度、资源分布），观察多物种在 AI 引导下的协同演化——自然选择、基因突变、物种形成、大灭绝。

### 核心特性

- **多物种生态系统** — 生产者、消费者、捕食者，食物网由基因涌现式决定
- **基因遗传系统** — 12 个基因控制体型、代谢类型、温度耐受、繁殖率等特征
- **气候与环境动力学** — 纬度梯度、季节循环、温室效应、火山热点
- **AI 驱动演化** — 大语言模型决定突变方向和物种形成（OpenAI / Claude / 自定义）
- **Web 前端** — React 实时渲染，Canvas 热力图 + Chart.js 数据图表
- **存档系统** — 手动存档、自动存档、读档恢复完整模拟状态
- **自定义星球** — 自由设定温度、重力、大气、轨道等所有物理参数
- **数据导出** — SQLite 持久化，CSV / JSON 导出

### 预设星球

| 星球 | 温度 | 特点 |
|------|------|------|
| Titan (土卫六) | -179°C | 甲烷湖泊、氮气大气、低重力 |
| Mars (火星) | -60°C | 稀薄 CO₂ 大气、高辐射 |
| Europa (木卫二) | -160°C | 冰壳下海洋、潮汐加热 |
| Kepler-442b | -2°C | 类地行星、可能有液态水 |
| 自定义 | 任意 | 自由设定所有物理参数 |

## 快速开始

### 环境要求

- Python 3.11+

### 安装

```bash
cd backend
pip install -r requirements.txt
```

### 配置 AI（可选）

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

支持的服务商：
- **OpenAI** — `OPENAI_API_KEY` + `OPENAI_MODEL`
- **Claude** — `ANTHROPIC_API_KEY` + `CLAUDE_MODEL`
- **自定义** — `CUSTOM_API_KEY` + `CUSTOM_API_BASE_URL` + `CUSTOM_MODEL`（DeepSeek、Moonshot、Ollama、vLLM 等 OpenAI 兼容接口）

### 启动

```bash
# Web 模式（默认），自动打开浏览器
python run_sim.py

# matplotlib 命令行模式
python run_sim.py --cli

# 自定义端口
python run_sim.py --port 3000
```

## Web 前端

启动后浏览器自动打开 `http://localhost:8000`：

- **新建模拟** — 选择星球（或自定义）、设置物种数量、选择 AI 引擎
- **实时渲染** — Canvas 热力图显示生物量 / 温度 / 资源分布
- **数据图表** — 种群曲线、Shannon 多样性指数、温度和 CO₂ 变化
- **控制面板** — 播放 / 暂停 / 单步 / 速度调节（1x - 100x）
- **存档系统** — 手动存档、读档、删除，支持自动存档
- **数据导出** — 侧边栏一键导出 CSV / JSON
- **历史记录** — 查看所有模拟运行记录

## 项目结构

```
backend/
├── run_sim.py                 # 启动入口（Web / CLI）
├── server.py                  # Web 服务器
├── visualize.py               # matplotlib 可视化
├── .env.example               # AI 配置模板
├── simulation/
│   ├── models.py              # 数据模型 (Gene, Genome, MetabolicType)
│   ├── genetics.py            # 突变、交叉、遗传距离
│   ├── environment.py         # 星球环境 (温度、资源、光照、大气)
│   ├── species.py             # 物种模型 (空间生物质分布)
│   ├── food_web.py            # 涌现式食物网
│   ├── engine.py              # 模拟引擎主循环 + AI 集成
│   └── templates.py           # JSON 配置加载器
├── ai/
│   ├── provider.py            # AI 抽象接口
│   ├── prompts.py             # Prompt 模板
│   ├── openai_provider.py     # OpenAI 实现
│   ├── claude_provider.py     # Claude 实现
│   └── factory.py             # 工厂函数（从 .env 加载配置）
├── api/
│   ├── app.py                 # FastAPI (REST + WebSocket)
│   └── sim_manager.py         # 模拟管理器 (后台线程 + 广播)
├── data/
│   ├── database.py            # SQLite 持久化
│   └── saves.py               # 存档系统
├── frontend/
│   └── index.html             # React 单页前端
├── config/
│   ├── planets/               # 星球配置 JSON
│   └── species/               # 物种模板 JSON
└── tests/                     # 单元测试
```

## 基因系统

每个物种由 12 个基因定义：

| 基因 | 说明 | 值域 |
|------|------|------|
| body_size | 体型 | 0.1 - 10.0 |
| metabolic_type | 代谢类型 | photosynthesis / chemosynthesis / heterotrophy |
| temp_optimum | 最适温度 | -250°C - 500°C |
| temp_tolerance | 温度耐受范围 | 5°C - 150°C |
| reproduction_rate | 繁殖率 | 0.01 - 3.0 |
| reproduction_cost | 繁殖消耗 | 0.1 - 5.0 |
| defense | 防御能力 | 0.0 - 1.0 |
| mobility | 移动速度 | 0.0 - 1.0 |
| sensory_range | 感知范围 | 1 - 15 格 |
| diet_preference | 食性 | producer / consumer / omni |
| lifespan | 寿命 | 10 - 2000 tick |
| adaptability | 适应性 | 0.0 - 1.0 |

食物关系由基因涌现式决定：光合生物是生产者，杂食者可吃任何生物，捕食效率 = 攻击力 / (攻击力 + 防御力)。

## API 端点

### 模拟控制
```
POST   /api/simulation/start      # 启动模拟
POST   /api/simulation/pause      # 暂停
POST   /api/simulation/resume     # 继续
POST   /api/simulation/stop       # 停止
POST   /api/simulation/speed      # 设置速度
```

### 存档
```
GET    /api/saves                 # 存档列表
POST   /api/saves                 # 手动存档
POST   /api/saves/load            # 读档
DELETE /api/saves/{id}            # 删除存档
POST   /api/simulation/auto-save  # 设置自动存档
```

### 数据
```
GET    /api/runs                  # 运行记录
GET    /api/runs/{id}/snapshots   # 快照数据
GET    /api/runs/{id}/species     # 物种历史
GET    /api/runs/{id}/events      # 事件日志
POST   /api/runs/{id}/export/csv  # 导出 CSV
POST   /api/runs/{id}/export/json # 导出 JSON
```

### 配置
```
GET    /api/config                # AI 配置（从 .env）
GET    /api/planets               # 可用星球列表
```

### WebSocket
```
ws://localhost:8000/ws/simulation  # 实时状态推送
```

## 技术栈

| 层 | 技术 |
|---|------|
| 模拟引擎 | Python, NumPy, SciPy |
| 数据模型 | Pydantic |
| 后端 API | FastAPI, Uvicorn |
| 实时通信 | WebSocket |
| 前端 | React 18 (CDN), Chart.js, HTML5 Canvas |
| 数据库 | SQLite |
| AI | OpenAI / Claude / 自定义 (可插拔) |
| 可视化 | Matplotlib (CLI 模式) |
| 测试 | pytest |

## 路线图

- [x] **Phase 1** — 核心模拟引擎（基因遗传、食物网、环境动力学）
- [x] **Phase 2** — AI 演化引擎（OpenAI / Claude / 自定义 Provider）
- [x] **Phase 3** — Web 前端（FastAPI + WebSocket + React Canvas）
- [x] **Phase 4** — 数据持久化（SQLite、存档系统、CSV/JSON 导出）
- [ ] **Phase 5** — 灾难事件系统、更多星球模板、性能优化

## 设计文档

- [设计规格文档](DESIGN_SPEC.md) — 架构、数据模型、API 设计
- [实现计划](IMPLEMENTATION_PLAN.md) — TDD 任务分解
