# LifeEvolutionSimulation

基于 AI 模拟生物在外星球极端环境上演化的研究平台。

## 项目简介

这是一个科学严谨的外星球生态系统演化模拟器。用户可以配置不同的星球参数（重力、大气、温度、资源分布），观察多物种在 AI 引导下的协同演化过程——包括自然选择、基因突变、物种形成和大灭绝。

### 核心特性

- **多物种生态系统** — 生产者、消费者、捕食者，食物网由基因涌现式决定
- **基因遗传系统** — 12 个基因控制体型、代谢类型、温度耐受、繁殖率等特征
- **气候与环境动力学** — 纬度梯度、季节循环、温室效应、火山热点
- **AI 驱动演化** — 大语言模型（OpenAI / Claude）决定突变方向和物种形成
- **实验框架** — A/B 对比、蒙特卡洛运行、时间线回放、数据导出
- **2D 实时可视化** — 生物量热力图、温度场、资源分布、种群曲线

### 预设星球

| 星球 | 温度 | 特点 |
|------|------|------|
| Titan (土卫六) | -179°C | 甲烷湖泊、氮气大气、低重力 |
| Mars (火星) | -60°C | 稀薄 CO2 大气、高辐射 |
| Europa (木卫二) | -160°C | 冰壳下海洋、潮汐加热 |
| Kepler-442b | -2°C | 类地行星、可能有液态水 |

## 快速开始

### 环境要求

- Python 3.11+

### 安装

```bash
cd backend
pip install -r requirements.txt
```

### 运行测试

```bash
python -m pytest tests/ -v
```

### 启动模拟

```bash
# 默认：Titan 星球，2 个生产者 + 1 个消费者
python run_sim.py

# 自定义参数
python run_sim.py --planet mars --producers 3 --consumers 2 --steps 5000

# 更大网格
python run_sim.py --planet europa --grid 100 --steps 10000
```

模拟启动后会弹出 matplotlib 窗口，实时显示 4 个面板：

1. **生物量热力图** — 各物种的空间分布
2. **温度场** — 星球表面温度分布（含季节变化）
3. **资源分布** — 可用资源的空间丰度
4. **种群曲线** — 各物种生物量随时间变化 + Shannon 多样性指数

按 `Ctrl+C` 随时停止。

## 项目结构

```
backend/
├── run_sim.py                 # 命令行入口
├── visualize.py               # matplotlib 可视化
├── simulation/
│   ├── models.py              # 数据模型 (Gene, Genome, MetabolicType)
│   ├── genetics.py            # 突变、交叉、遗传距离计算
│   ├── environment.py         # 星球环境 (温度、资源、光照、大气)
│   ├── species.py             # 物种模型 (空间生物质分布)
│   ├── food_web.py            # 涌现式食物网
│   ├── engine.py              # 模拟引擎主循环
│   └── templates.py           # JSON 配置加载器
├── config/
│   ├── planets/               # 星球配置 (titan, mars, europa, kepler442b)
│   └── species/               # 物种模板 (producer_photo, producer_chemo, consumer_herbivore)
└── tests/                     # 41 个单元测试
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

## 技术栈

| 层 | 技术 |
|---|------|
| 模拟引擎 | Python, NumPy, SciPy |
| 数据模型 | Pydantic |
| 可视化 | Matplotlib |
| 测试 | pytest |
| 后端 API | FastAPI (Phase 3) |
| 前端 | React + Canvas (Phase 3) |
| AI | OpenAI / Claude 可插拔 (Phase 2) |

## 路线图

- [x] **Phase 1** — 核心模拟引擎（基因遗传、食物网、环境动力学、CLI 可视化）
- [ ] **Phase 2** — AI 演化引擎（OpenAI/Claude Provider，智能突变决策）
- [ ] **Phase 3** — Web 前端（FastAPI + WebSocket + React Canvas 实时渲染）
- [ ] **Phase 4** — 实验框架（SQLite 持久化、时间线回放、A/B 对比分析）
- [ ] **Phase 5** — 灾难事件系统、UI 美化、性能优化

## 设计文档

详细设计请参阅：

- [设计规格文档](DESIGN_SPEC.md) — 完整的架构、数据模型、API 设计
- [实现计划](IMPLEMENTATION_PLAN.md) — Phase 1 的 TDD 任务分解
