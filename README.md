# LifeEvolutionSimulation

基于 AI 模拟生物在外星球极端环境上演化的研究平台。

## 项目简介

科学严谨的外星球生态系统演化模拟器。配置星球参数（重力、大气、温度、资源分布），观察多物种在 AI 引导下的协同演化——自然选择、基因突变、物种形成、大灭绝。

### 核心特性

- **多物种生态系统** — 生产者、消费者、捕食者，食物网由基因涌现式决定
- **基因遗传系统** — 12 个基因控制体型、代谢类型、温度耐受、繁殖率等特征
- **7 种生态交互** — 分解循环、大气反馈、生物热效应、毒素系统、承载力、互利共生、光照竞争
- **AI 驱动演化** — 大语言模型决定突变方向和物种形成（OpenAI / Claude / 自定义）
- **GPU 加速** — CuPy CUDA 加速网格运算，无 GPU 自动回退 NumPy
- **Web 前端** — React 实时渲染，Canvas 热力图 + Chart.js 数据图表
- **自定义星球** — 自由设定温度、重力、大气、轨道等所有物理参数
- **自定义物种** — 自由配置每个物种的 12 个基因，支持预设模板
- **存档系统** — 手动存档、自动存档、读档恢复完整模拟状态
- **数据持久化** — SQLite 记录，CSV / JSON 导出

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
- CUDA（可选，用于 GPU 加速）

### 安装

```bash
cd backend
pip install -r requirements.txt
```

### GPU 加速（可选）

```bash
pip install cupy-cuda12x   # CUDA 12.x
# 或
pip install cupy-cuda11x   # CUDA 11.x
```

无 CUDA 环境自动使用 CPU（NumPy），无需配置。可用 `DISABLE_GPU=1` 环境变量强制禁用。

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

启动后浏览器自动打开 `http://localhost:8000`。

### 界面布局

- **中央** — Canvas 实时渲染（生物量 / 温度 / 资源 / 毒素 / 营养热力图 + 物种光晕）
- **右侧栏** — 环境参数（温度、资源、大气成分进度条、交互指标、数据管理）
- **底部面板** — 生物数据（物种列表、种群统计、基因概览、事件日志）

### 功能

- **新建模拟** — 选择星球（或自定义）、添加物种（6 种预设或自定义基因）、选择 AI 引擎
- **自定义物种** — 展开编辑 12 个基因滑块：代谢类型、食性、体型、温度、繁殖、防御、移动等
- **自定义星球** — 设定表面温度、重力、反照率、转轴倾角、轨道距离、大气压、CO₂/CH₄ 比例等
- **实时控制** — 播放 / 暂停 / 单步 / 速度（1x - 100x）
- **图层切换** — 生物量、温度、资源、毒素、营养 5 种热力图
- **存档系统** — 手动存档、读档、删除，支持每 500 步自动存档
- **数据导出** — CSV / JSON 一键导出

### 可视化增强

- 双线性插值热力图（平滑过渡）
- HSL 多段渐变色（自适应数据范围）
- 物种光晕渲染（径向渐变 + 白色高亮核心）
- 高分屏 DPI 适配
- 左下角图例（自动缩放数值范围）

## 生态交互机制

模拟引擎实现了 7 种生物-环境交互：

1. **分解循环** — 死亡生物量进入分解池，缓慢释放回资源和营养（每 tick 5% 衰减）
2. **大气反馈** — 异养生物需要 O₂，高 CO₂ 产生压力；光合生物受益于高 CO₂；化能生物释放 CH₄
3. **生物热效应** — 密集生物群落产生局部热量，消费者产热更多
4. **毒素系统** — 异养生物排泄废物产生毒素，毒素扩散并缓慢衰减，适应性强的物种有抗性
5. **环境承载力** — 种群密度越高，增长越慢（逻辑斯谛限制）
6. **互利共生** — 消费者废物为生产者提供营养加成；生产者 O₂ 为消费者提供适应度加成
7. **光照竞争** — 大型生物遮蔽小型光合生物的光照

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
│   ├── environment.py         # 星球环境 (GPU 加速)
│   ├── species.py             # 物种模型 (多样化颜色)
│   ├── food_web.py            # 涌现式食物网
│   ├── engine.py              # 模拟引擎 (7 种交互 + AI)
│   ├── gpu_backend.py         # GPU 后端 (CuPy/NumPy 自动切换)
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
GET    /api/config                # AI + GPU 配置（从 .env）
GET    /api/planets               # 可用星球列表
```

### WebSocket
```
ws://localhost:8000/ws/simulation  # 实时状态推送（含网格数据）
```

## 技术栈

| 层 | 技术 |
|---|------|
| 模拟引擎 | Python, NumPy/SciPy, CuPy (GPU) |
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
- [x] **Phase 5** — 生态交互增强（分解、大气、毒素、共生、承载力、光照竞争）
- [x] **Phase 6** — GPU 加速、自定义星球/物种、UI 优化
- [ ] **Phase 7** — 灾难事件系统、更多星球模板、性能优化

## 设计文档

- [设计规格文档](DESIGN_SPEC.md) — 架构、数据模型、API 设计
- [实现计划](IMPLEMENTATION_PLAN.md) — TDD 任务分解
