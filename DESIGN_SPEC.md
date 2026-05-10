# 外星球生物演化模拟器 — 设计规格文档

> **项目名称**: LifeEvolutionSimulation
> **版本**: v2.0 (完整重设计)
> **日期**: 2026-05-10
> **目标**: 基于 AI 的外星球生态系统演化模拟研究平台

---

## 1. 项目概述

### 1.1 愿景
构建一个科学严谨、功能完整的外星球生物演化模拟平台。用户可以配置星球参数（重力、大气、温度、资源分布），观察多物种在 AI 引导下的协同演化过程，支持实验对比、数据导出和回放分析。

### 1.2 核心特性
- **多物种生态系统**: 生产者、消费者、捕食者、分解者 — 涌现式食物网
- **基因遗传系统**: DNA 编码、突变、基因表达、自然选择
- **气候与环境动力学**: 季节循环、温室效应、灾难事件
- **AI 驱动演化**: 大语言模型决定突变方向、物种形成、大灭绝响应
- **实验框架**: A/B 对比、蒙特卡洛运行、时间线回放、数据导出
- **2D 实时可视化**: Canvas 渲染生物体、热力图环境层、交互式仪表板

### 1.3 技术栈
| 层 | 技术 | 说明 |
|---|------|------|
| 前端 | React + TypeScript | UI 框架 |
| 可视化 | HTML5 Canvas + Recharts | 2D 世界渲染 + 数据图表 |
| 后端 | Python FastAPI | 模拟引擎 + API 服务 |
| AI | OpenAI / Claude (可插拔) | 演化决策 |
| 数据 | SQLite + JSON | 实验存储 + 配置文件 |
| 通信 | WebSocket + REST | 实时数据流 + 控制接口 |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    React 前端                            │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐  │
│  │ 2D 世界  │  │ 数据图表 │  │ 实验控制  │  │ 对比   │  │
│  │ (Canvas) │  │(Recharts)│  │   面板    │  │ 仪表板 │  │
│  └──────────┘  └──────────┘  └───────────┘  └────────┘  │
│         WebSocket (实时数据) + REST API (控制)            │
├─────────────────────────────────────────────────────────┤
│                  Python FastAPI 后端                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐  │
│  │ 模拟引擎 │  │ AI 层    │  │ 实验管理  │  │ 数据   │  │
│  │ 生态系统 │  │ (可插拔  │  │ (保存/    │  │ 持久化 │  │
│  │ 基因遗传 │  │ OpenAI/  │  │  对比/    │  │ SQLite │  │
│  │ 环境动力 │  │  Claude) │  │  回放)    │  │        │  │
│  └──────────┘  └──────────┘  └───────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.1 后端模块结构
```
backend/
├── main.py                  # FastAPI 入口
├── simulation/
│   ├── engine.py            # 模拟主循环
│   ├── ecosystem.py         # 多物种生态系统
│   ├── genetics.py          # DNA 编码与遗传
│   ├── environment.py       # 环境层与气候动力学
│   ├── food_web.py          # 食物网动态
│   └── disasters.py         # 灾难事件系统
├── ai/
│   ├── provider.py          # AI 抽象接口
│   ├── openai_provider.py   # OpenAI 实现
│   ├── claude_provider.py   # Claude 实现
│   └── prompts.py           # Prompt 模板
├── experiments/
│   ├── manager.py           # 实验管理器
│   ├── recorder.py          # 时间线记录器
│   └── comparator.py        # 对比分析
├── api/
│   ├── routes.py            # REST 端点
│   └── websocket.py         # WebSocket 处理
├── data/
│   ├── models.py            # 数据模型 (Pydantic)
│   └── database.py          # SQLite 操作
└── config/
    ├── planets/             # 星球配置 JSON
    └── species/             # 物种模板 JSON
```

### 2.2 前端模块结构
```
frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── WorldCanvas/     # 2D 世界渲染
│   │   │   ├── Canvas.tsx
│   │   │   ├── OrganismRenderer.tsx
│   │   │   └── EnvironmentLayer.tsx
│   │   ├── Dashboard/       # 数据仪表板
│   │   │   ├── PopulationChart.tsx
│   │   │   ├── DiversityChart.tsx
│   │   │   └── EnvironmentChart.tsx
│   │   ├── ControlPanel/    # 控制面板
│   │   │   ├── SimulationControls.tsx
│   │   │   ├── DisasterTriggers.tsx
│   │   │   └── ParameterEditor.tsx
│   │   ├── Experiment/      # 实验管理
│   │   │   ├── ExperimentCreator.tsx
│   │   │   ├── RunSelector.tsx
│   │   │   └── ComparisonView.tsx
│   │   └── Timeline/        # 时间线回放
│   │       └── TimelineScrubber.tsx
│   ├── hooks/
│   │   ├── useSimulation.ts # WebSocket 连接
│   │   └── useExperiment.ts # 实验状态管理
│   ├── services/
│   │   └── api.ts           # REST API 客户端
│   └── types/
│       └── index.ts         # TypeScript 类型定义
```

---

## 3. 核心数据模型

### 3.1 基因组 (Genome)
```python
class Genome:
    genes: dict[str, Gene]  # 基因名 → Gene 对象
    generation: int          # 世代数
    parent_ids: list[str]    # 亲本 ID

class Gene:
    value: float             # 当前值
    dominance: float         # 显性程度 (0-1)
    mutation_rate: float     # 突变概率
    min_value: float         # 值域下限
    max_value: float         # 值域上限
```

**基因列表:**

| 基因名 | 类型 | 值域 | 影响 |
|--------|------|------|------|
| body_size | float | 0.1 - 10.0 | 体型 → 资源需求、速度、被捕食概率 |
| metabolic_type | enum | photo/chemo/hetero | 能量获取方式 |
| temp_optimum | float | -250 - 500 | 最适温度 |
| temp_tolerance | float | 5 - 150 | 温度耐受范围 (°C) |
| reproduction_rate | float | 0.01 - 3.0 | 每周期繁殖率 |
| reproduction_cost | float | 0.1 - 5.0 | 繁殖资源消耗 |
| defense | float | 0.0 - 1.0 | 防御能力 |
| mobility | float | 0.0 - 1.0 | 移动速度 |
| sensory_range | int | 1 - 15 | 感知范围 (格) |
| diet_preference | enum | producer/consumer/omni | 食性 |
| lifespan | int | 10 - 2000 | 寿命 (tick) |
| adaptability | float | 0.0 - 1.0 | 基因突变方向的有利概率 |

### 3.2 物种 (Species)
```python
class Species:
    id: str                    # 唯一标识
    name: str                  # AI 生成的名称
    genome: Genome             # 基因组
    population: ndarray        # 50x50 空间分布 (每个格子的个体数)
    biomass: ndarray           # 50x50 生物量分布
    fitness: ndarray           # 50x50 适应度
    ancestor_id: str | None    # 祖先物种 ID
    extinction_tick: int | None  # 灭绝时间 (若已灭绝)
    color: tuple[int,int,int]  # 可视化颜色
    history: list[EvolutionEvent]  # 演化事件记录
```

### 3.3 环境状态 (EnvironmentState)
```python
class EnvironmentState:
    temperature: ndarray       # 50x50 温度场 (°C)
    resources: ndarray         # 50x50 资源丰度 (0-2)
    light: ndarray             # 50x50 光照强度
    water: ndarray             # 50x50 液态水/溶剂
    atmosphere: dict           # {O2, CO2, CH4, N2, Pressure}
    tick: int                  # 当前时间步
    season: str                # 当前季节
```

### 3.4 实验 (Experiment)
```python
class Experiment:
    id: str
    name: str
    description: str
    planet_config: PlanetConfig
    shared_params: SimParams
    created_at: datetime
    runs: list[ExperimentRun]

class ExperimentRun:
    id: str
    experiment_id: str
    seed: int
    status: enum  # pending/running/completed/failed
    timeline: list[Snapshot]  # 每 N tick 的快照
    final_stats: RunStats
    ai_decisions: list[AIDecision]
```

---

## 4. 模拟引擎

### 4.1 主循环
```
每 tick:
  1. 更新环境层
     - 温度: 纬度梯度 + 季节偏移 + 温室效应 + 随机扰动
     - 资源: 自然再生 - 消耗 + 分解者回收
     - 光照: 太阳距离 + 大气遮蔽 + 季节变化
     - 大气: 由物种代谢产物累积

  2. 计算各物种适应度
     - 温度适应度 = gaussian(temp - optimum, tolerance)
     - 资源可得性 = local_resources / resource_need
     - 竞争压力 = sum(competitors_in_cell)
     - 捕食压力 = sum(predators_in_cell * attack / defense)
     - fitness = product(上述因子) * random_noise

  3. 物种动态
     - 出生: population * reproduction_rate * fitness * resources
     - 死亡: population * (base_mortality + starvation + predation + env_stress)
     - 移动: 按 mobility 基因扩散到邻近格子
     - 捕食: predator.biomass += prey.biomass * predation_rate

  4. 环境反馈
     - 光合生物: 释放 O2, 吸收 CO2
     - 异养生物: 吸收 O2, 释放 CO2
     - 化能生物: 释放 CH4, H2S
     - 所有生物: 产热 (微量)

  5. AI 演化决策 (每 N tick)
     - 检测: 压力物种 (适应度持续下降)、新兴机会 (资源富集区空缺)
     - AI 输出: 突变方向、新物种参数、环境叙事
     - 执行: 应用基因突变、创建新物种分支

  6. 记录快照 (每 M tick)
     - 物种状态、环境状态、事件日志
```

### 4.2 食物网动态
食物关系不是硬编码的，而是由基因决定的涌现行为:
- `metabolic_type` = photosynthesis → 生产者 (不吃任何生物)
- `metabolic_type` = heterotrophy + `diet_preference` = producer → 初级消费者
- `metabolic_type` = heterotrophy + `diet_preference` = consumer → 次级消费者
- `diet_preference` = omni → 可吃生产者和消费者 (效率较低)
- 死亡生物量 → 分解者回收

捕食效率 = attack_power / (attack_power + prey_defense)
attack_power = body_size * mobility * sensory_range

### 4.3 突变机制
自然突变: 每个 tick，每个基因有 `mutation_rate` 概率发生小幅度随机变化 (±5%)
AI 引导突变: 每 N tick，AI 分析生态系统状态，为压力物种推荐有方向性的突变
物种形成: 当某个种群的基因组与祖先差异超过阈值，且地理隔离足够，AI 可触发物种形成事件

---

## 5. 环境与气候系统

### 5.1 温度模型
```
T(lat, lon, tick) = base_temp(lat)
                   + seasonal_offset(axial_tilt, tick)
                   + greenhouse_effect(CO2, CH4)
                   + volcanic_heat(lat, lon)
                   + biological_heat(biomass)
                   + random_perturbation
```

- **纬度梯度**: 赤道热、两极冷，由 `distance` 和 `albedo` 决定基础温度
- **季节循环**: `axial_tilt` 决定振幅，周期可配置
- **温室效应**: CO2 和 CH4 浓度 → 温度增量 (对数关系)
- **火山热**: 随机分布的热点，缓慢衰减
- **生物热**: 密集生物群落微量产热

### 5.2 资源系统
- **可再生资源**: 有机物，基础再生率 + 光照促进
- **不可再生资源**: 矿物质，缓慢消耗，火山事件可补充
- **空间分布**: 初始分布从 Excel/JSON 加载，运行时动态变化

### 5.3 灾难事件
| 事件 | 影响范围 | 持续时间 | 效果 |
|------|----------|----------|------|
| 小行星撞击 | 局部 (半径 R) | 瞬时 + 余波 | 摧毁生物量、注入尘埃 (降温)、释放矿物质 |
| 太阳耀斑 | 全球 | 短期 (10-50 tick) | 辐射增加、损害表面生物 |
| 超级火山 | 区域 + 全球大气 | 长期 (100-500 tick) | 注入气体、改变大气组成 |
| 冰河期 | 全球 | 极长期 (1000+ tick) | 温度持续下降 |
| 海洋酸化 | 水域 | 中期 | CO2 溶入水体，影响水生生物 |

---

## 6. AI 演化引擎

### 6.1 可插拔 Provider 接口
```python
class AIProvider(ABC):
    @abstractmethod
    async def analyze_ecosystem(self, state: EcosystemSnapshot) -> EvolutionDecision:
        """分析生态系统状态，返回演化决策"""
        pass

    @abstractmethod
    async def generate_narrative(self, event: str, context: dict) -> str:
        """为演化事件生成教育性描述"""
        pass

    @abstractmethod
    async def name_species(self, traits: dict) -> str:
        """根据特征为新物种命名"""
        pass
```

### 6.2 AI 决策输出格式
```json
{
  "mutations": [
    {
      "species_id": "sp_001",
      "gene": "temp_tolerance",
      "delta": 5.0,
      "reason": "种群面临温度波动压力，扩大耐受范围"
    }
  ],
  "speciation": [
    {
      "parent_species_id": "sp_003",
      "new_species": {
        "name": "嗜热甲烷菌",
        "genome_overrides": {
          "temp_optimum": 80.0,
          "metabolic_type": "chemosynthesis"
        }
      },
      "reason": "火山口附近资源丰富但温度极高，催生新生态位"
    }
  ],
  "narrative": "在第 500 个周期，星球北半球进入严冬。嗜冷种群..."
}
```

### 6.3 Prompt 策略
- **系统 Prompt**: 设定 AI 为"外星生物学家"角色，描述星球物理规则
- **上下文**: 当前环境摘要、物种列表及适应度、最近事件
- **约束**: 突变必须符合物理合理性 (不能在 -200°C 突变出需要液态水的生物)
- **频率**: 默认每 60 tick 调用一次，可配置

---

## 7. 实验框架

### 7.1 实验类型
1. **单次运行**: 配置参数 → 运行 → 观察
2. **A/B 对比**: 同星球不同参数 (如重力 1g vs 2g)
3. **多星球对比**: 同初始物种，不同星球环境
4. **蒙特卡洛**: 同配置不同随机种子，统计显著性

### 7.2 数据记录
每 N tick 记录快照:
- 所有物种的种群数、平均适应度、基因频率
- 环境指标 (全球平均温度、大气组成、总资源量)
- 事件日志 (突变、物种形成、灭绝、灾难)
- AI 决策及其推理过程

### 7.3 对比分析
- 叠加图表: 多次运行的同一指标画在同一坐标系
- 统计摘要: 均值、标准差、最大最小值
- 显著性检验: 两组运行结果是否有统计差异

### 7.4 回放系统
- 时间线滑块: 拖动到任意 tick 查看状态
- 播放控制: 暂停、步进、加速
- 事件标记: 在时间线上标记重要事件

---

## 8. 前端可视化

### 8.1 2D 世界渲染 (Canvas)
- **底层**: 环境热力图 (温度/资源/光照/大气，可切换)
- **中层**: 生物体渲染
  - 大小 ∝ body_size 基因
  - 颜色: 绿色系=光合, 红色系=异养, 蓝色系=化能
  - 形状复杂度 ∝ 演化复杂度 (圆点 → 不规则形状)
  - 动画: 移动方向和速度反映 mobility 基因
- **顶层**: 食物网连接线 (可选显示)

### 8.2 数据仪表板
- **种群图**: 堆叠面积图，每种物种一个色带
- **多样性指数**: 折线图，Shannon-Wiener 指数
- **环境指标**: 温度、大气组成的实时曲线
- **事件时间线**: 标记重要演化事件

### 8.3 控制面板
- 播放/暂停/步进
- 速度控制 (1x, 5x, 10x, 50x, 100x)
- 手动触发灾难
- 实时调整环境参数
- 物种查看器 (点击物种看详细基因组)

---

## 9. API 设计

### 9.1 REST 端点
```
POST   /api/simulation/start          # 启动新模拟
POST   /api/simulation/pause          # 暂停
POST   /api/simulation/step           # 单步
POST   /api/simulation/resume         # 继续
POST   /api/simulation/stop           # 停止
PUT    /api/simulation/params         # 修改运行参数

GET    /api/simulation/state          # 获取当前状态
GET    /api/simulation/species        # 物种列表
GET    /api/simulation/species/{id}   # 物种详情
GET    /api/simulation/environment    # 环境状态

POST   /api/disaster/trigger          # 触发灾难
POST   /api/evolution/force           # 强制 AI 演化决策

GET    /api/experiments               # 实验列表
POST   /api/experiments               # 创建实验
GET    /api/experiments/{id}          # 实验详情
GET    /api/experiments/{id}/runs     # 运行列表
POST   /api/experiments/{id}/runs     # 启动新运行
GET    /api/experiments/{id}/compare  # 对比分析

GET    /api/planets                   # 可用星球列表
GET    /api/planets/{name}            # 星球配置
POST   /api/planets                   # 上传自定义星球
```

### 9.2 WebSocket 通道
```
ws://localhost:8000/ws/simulation
→ 每 tick 推送: { tick, species_summary, environment_summary, events }
→ 帧率: 根据速度设置动态调整
```

---

## 10. 默认星球与物种模板

### 10.1 预设星球
1. **Titan (土卫六)**: -179°C, 甲烷湖泊, 氮气大气, 低重力
2. **Mars (火星)**: -60°C, 稀薄 CO2 大气, 高辐射
3. **Europa (木卫二)**: -160°C, 冰壳下液态海洋, 潮汐加热
4. **Kepler-442b**: 类地行星, 可能有液态水
5. **自定义**: 用户完全自定义所有参数

### 10.2 初始物种模板
- **硅基光合体**: 化能自养, 耐高温, 用于火山星球
- **甲烷代谢体**: 甲烷溶剂, 耐极寒, 用于 Titan 类星球
- **水基微生物**: 水溶剂, 中温范围, 用于类地行星
- **辐射食者**: 直接利用辐射能, 耐高辐射环境

---

## 11. 非功能需求

### 11.1 性能
- 50x50 网格, 20+ 物种: 实时模拟 ≥ 10 tick/s
- WebSocket 延迟 < 100ms
- Canvas 渲染 60fps (不包含模拟计算)

### 11.2 可扩展性
- 网格大小可配置 (最高 200x200)
- 物种数量无硬限制 (受计算资源约束)
- AI Provider 可扩展 (添加新 LLM 只需实现接口)

### 11.3 数据完整性
- 实验数据持久化到 SQLite
- 时间线快照支持完整回放
- 导出格式: CSV, JSON

---

## 12. 实现优先级

### Phase 1: 核心模拟引擎
- 基因遗传系统
- 多物种生态系统 (食物网)
- 环境层与气候动力学
- 命令行可视化 (matplotlib, 延续现有代码)

### Phase 2: AI 集成
- Provider 接口 + OpenAI 实现
- AI 演化决策逻辑
- 物种命名与叙事生成
- Claude Provider 实现

### Phase 3: Web 前端
- FastAPI 后端 + WebSocket
- React 前端骨架
- Canvas 2D 世界渲染
- 数据仪表板

### Phase 4: 实验框架
- 实验管理器
- 时间线记录与回放
- 对比分析
- 数据导出

### Phase 5: 完善与优化
- 灾难事件系统
- 预设星球与物种模板
- UI 美化与交互优化
- 性能优化

---

## 附录 A: 与现有代码的关系

现有 `main.py` 中的 `PlanetEnvironmentLoader` 将被重构为 `environment.py` 的一部分。Excel 加载逻辑保留，但增加 JSON 配置支持。`CoevolutionSim` 的核心逻辑将拆分为 `ecosystem.py`, `genetics.py`, `food_web.py`。matplotlib 可视化在 Phase 1 保留，Phase 3 被 Web 前端替代。
