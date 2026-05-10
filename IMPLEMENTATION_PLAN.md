# Phase 1 实现计划：核心模拟引擎

> **前置条件**: 已完成 DESIGN_SPEC.md 审阅和批准
> **目标**: 构建完整的多物种生态系统模拟引擎，含基因遗传、食物网、环境动力学
> **验证**: 命令行运行 matplotlib 可视化，观察多物种在 Titan 环境下协同演化

---

## Task 1: 项目结构与依赖

### 目标
搭建后端项目骨架，安装依赖，配置测试框架。

### 文件
- `backend/requirements.txt` (新建)
- `backend/pyproject.toml` (新建)
- `backend/tests/__init__.py` (新建)
- `backend/tests/conftest.py` (新建)

### 步骤

**1.1 创建 requirements.txt**

```txt
# backend/requirements.txt
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
websockets>=11.0
openai>=1.0.0
anthropic>=0.18.0
aiosqlite>=0.19.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

**1.2 创建 pyproject.toml**

```toml
# backend/pyproject.toml
[project]
name = "life-evolution-sim"
version = "2.0.0"
description = "AI-driven alien planet evolution simulator"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["."]
```

**1.3 创建测试配置**

```python
# backend/tests/__init__.py
# empty
```

```python
# backend/tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def small_grid_size():
    """Small grid for fast tests."""
    return 10

@pytest.fixture
def default_grid_size():
    """Standard grid size."""
    return 50
```

**1.4 验证**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
pip install -r requirements.txt --break-system-packages
python -m pytest --co -q
```

预期输出: `no tests ran` (还没有测试文件)

### Commit
```
git add -A && git commit -m "chore: scaffold backend project structure with dependencies"
```

---

## Task 2: 数据模型 (Pydantic)

### 目标
定义所有核心数据类型。这些是整个系统的"语言"，后续每个模块都依赖它们。

### 文件
- `backend/simulation/__init__.py` (新建)
- `backend/simulation/models.py` (新建)
- `backend/tests/test_models.py` (新建)

### 步骤

**2.1 写失败测试**

```python
# backend/tests/test_models.py
from simulation.models import Gene, Genome, MetabolicType, DietPreference

class TestGene:
    def test_create_float_gene(self):
        gene = Gene(value=0.5, min_value=0.0, max_value=1.0, mutation_rate=0.05)
        assert gene.value == 0.5
        assert gene.min_value == 0.0
        assert gene.max_value == 1.0

    def test_gene_clamps_value(self):
        gene = Gene(value=1.5, min_value=0.0, max_value=1.0)
        assert gene.value == 1.0

    def test_gene_clamps_negative(self):
        gene = Gene(value=-0.5, min_value=0.0, max_value=1.0)
        assert gene.value == 0.0

class TestGenome:
    def test_create_genome_with_defaults(self):
        genome = Genome.create_default()
        assert "body_size" in genome.genes
        assert "temp_optimum" in genome.genes
        assert genome.generation == 0

    def test_genome_has_all_required_genes(self):
        genome = Genome.create_default()
        required = [
            "body_size", "metabolic_type", "temp_optimum", "temp_tolerance",
            "reproduction_rate", "reproduction_cost", "defense", "mobility",
            "sensory_range", "diet_preference", "lifespan", "adaptability"
        ]
        for gene_name in required:
            assert gene_name in genome.genes, f"Missing gene: {gene_name}"

    def test_metabolic_type_is_enum(self):
        assert MetabolicType.PHOTOSYNTHESIS.value == "photosynthesis"
        assert MetabolicType.CHEMOSYNTHESIS.value == "chemosynthesis"
        assert MetabolicType.HETEROTROPHY.value == "heterotrophy"

    def test_diet_preference_is_enum(self):
        assert DietPreference.PRODUCER.value == "producer"
        assert DietPreference.CONSUMER.value == "consumer"
        assert DietPreference.OMNI.value == "omni"
```

**2.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_models.py -v
```

预期输出: `FAILED - ModuleNotFoundError: No module named 'simulation.models'`

**2.3 实现数据模型**

```python
# backend/simulation/__init__.py
# empty
```

```python
# backend/simulation/models.py
from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator


class MetabolicType(str, Enum):
    PHOTOSYNTHESIS = "photosynthesis"
    CHEMOSYNTHESIS = "chemosynthesis"
    HETEROTROPHY = "heterotrophy"


class DietPreference(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    OMNI = "omni"


class Gene(BaseModel):
    value: float
    min_value: float
    max_value: float
    mutation_rate: float = 0.05
    dominance: float = 0.5

    @model_validator(mode="after")
    def clamp_value(self) -> "Gene":
        self.value = max(self.min_value, min(self.max_value, self.value))
        return self


class Genome(BaseModel):
    genes: dict[str, Gene | str]  # str for enum genes (metabolic_type, diet_preference)
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)

    def get_float(self, name: str) -> float:
        gene = self.genes[name]
        if isinstance(gene, Gene):
            return gene.value
        raise TypeError(f"Gene '{name}' is not a float gene")

    def get_enum(self, name: str) -> str:
        gene = self.genes[name]
        if isinstance(gene, str):
            return gene
        raise TypeError(f"Gene '{name}' is not an enum gene")

    @staticmethod
    def create_default() -> Genome:
        return Genome(
            genes={
                "body_size": Gene(value=1.0, min_value=0.1, max_value=10.0, mutation_rate=0.03),
                "metabolic_type": MetabolicType.PHOTOSYNTHESIS.value,
                "temp_optimum": Gene(value=20.0, min_value=-250.0, max_value=500.0, mutation_rate=0.02),
                "temp_tolerance": Gene(value=30.0, min_value=5.0, max_value=150.0, mutation_rate=0.02),
                "reproduction_rate": Gene(value=0.5, min_value=0.01, max_value=3.0, mutation_rate=0.03),
                "reproduction_cost": Gene(value=1.0, min_value=0.1, max_value=5.0, mutation_rate=0.02),
                "defense": Gene(value=0.1, min_value=0.0, max_value=1.0, mutation_rate=0.04),
                "mobility": Gene(value=0.3, min_value=0.0, max_value=1.0, mutation_rate=0.03),
                "sensory_range": Gene(value=3.0, min_value=1.0, max_value=15.0, mutation_rate=0.03),
                "diet_preference": DietPreference.PRODUCER.value,
                "lifespan": Gene(value=500.0, min_value=10.0, max_value=2000.0, mutation_rate=0.02),
                "adaptability": Gene(value=0.3, min_value=0.0, max_value=1.0, mutation_rate=0.04),
            }
        )

    @staticmethod
    def create_consumer() -> Genome:
        genome = Genome.create_default()
        genome.genes["metabolic_type"] = MetabolicType.HETEROTROPHY.value
        genome.genes["diet_preference"] = DietPreference.CONSUMER.value
        genome.genes["body_size"] = Gene(value=2.0, min_value=0.1, max_value=10.0, mutation_rate=0.03)
        genome.genes["mobility"] = Gene(value=0.6, min_value=0.0, max_value=1.0, mutation_rate=0.03)
        genome.genes["defense"] = Gene(value=0.2, min_value=0.0, max_value=1.0, mutation_rate=0.04)
        return genome

    @staticmethod
    def create_decomposer() -> Genome:
        genome = Genome.create_default()
        genome.genes["metabolic_type"] = MetabolicType.HETEROTROPHY.value
        genome.genes["diet_preference"] = DietPreference.CONSUMER.value
        genome.genes["body_size"] = Gene(value=0.2, min_value=0.1, max_value=10.0, mutation_rate=0.03)
        genome.genes["mobility"] = Gene(value=0.1, min_value=0.0, max_value=1.0, mutation_rate=0.03)
        genome.genes["reproduction_rate"] = Gene(value=1.5, min_value=0.01, max_value=3.0, mutation_rate=0.03)
        return genome
```

**2.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_models.py -v
```

预期输出: `4 passed`

### Commit
```
git add -A && git commit -m "feat: add Pydantic data models for Gene, Genome, MetabolicType, DietPreference"
```

---

## Task 3: 遗传系统 — 突变与继承

### 目标
实现基因突变、基因组复制与变异。这是演化的基础机制。

### 文件
- `backend/simulation/genetics.py` (新建)
- `backend/tests/test_genetics.py` (新建)

### 步骤

**3.1 写失败测试**

```python
# backend/tests/test_genetics.py
import numpy as np
from simulation.models import Genome, Gene, MetabolicType
from simulation.genetics import mutate_genome, crossover_genomes, calculate_genetic_distance

class TestMutation:
    def test_mutation_changes_value(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        original_val = original.get_float("body_size")

        # Mutate many times — at least one should differ
        mutated_any = False
        for _ in range(100):
            mutated = mutate_genome(original, rng=rng)
            if mutated.get_float("body_size") != original_val:
                mutated_any = True
                break
        assert mutated_any, "Mutation never changed body_size in 100 attempts"

    def test_mutation_respects_bounds(self):
        rng = np.random.RandomState(42)
        genome = Genome.create_default()
        for _ in range(1000):
            mutated = mutate_genome(genome, rng=rng)
            val = mutated.get_float("body_size")
            assert 0.1 <= val <= 10.0

    def test_mutation_increments_generation(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        mutated = mutate_genome(original, rng=rng)
        assert mutated.generation == original.generation + 1

    def test_mutation_preserves_parent_id(self):
        rng = np.random.RandomState(42)
        original = Genome.create_default()
        mutated = mutate_genome(original, parent_id="sp_001", rng=rng)
        assert "sp_001" in mutated.parent_ids

class TestCrossover:
    def test_crossover_produces_valid_genome(self):
        rng = np.random.RandomState(42)
        parent_a = Genome.create_default()
        parent_b = Genome.create_consumer()
        child = crossover_genomes(parent_a, parent_b, rng=rng)
        assert "body_size" in child.genes
        assert "metabolic_type" in child.genes
        assert child.generation == max(parent_a.generation, parent_b.generation) + 1

    def test_crossover_child_is_different_from_parents(self):
        rng = np.random.RandomState(42)
        parent_a = Genome.create_default()
        parent_b = Genome.create_consumer()
        child = crossover_genomes(parent_a, parent_b, rng=rng)
        # At least some float genes should differ from both parents
        differs = False
        for name in ["body_size", "temp_optimum", "defense"]:
            child_val = child.get_float(name)
            a_val = parent_a.get_float(name)
            b_val = parent_b.get_float(name)
            if child_val != a_val or child_val != b_val:
                differs = True
                break
        # Enum genes may also differ
        if child.get_enum("metabolic_type") != parent_a.get_enum("metabolic_type"):
            differs = True
        assert differs

class TestGeneticDistance:
    def test_identical_genomes_have_zero_distance(self):
        g = Genome.create_default()
        assert calculate_genetic_distance(g, g) == 0.0

    def test_different_genomes_have_positive_distance(self):
        a = Genome.create_default()
        b = Genome.create_consumer()
        dist = calculate_genetic_distance(a, b)
        assert dist > 0.0
```

**3.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_genetics.py -v
```

预期输出: `FAILED - ModuleNotFoundError: No module named 'simulation.genetics'`

**3.3 实现遗传系统**

```python
# backend/simulation/genetics.py
from __future__ import annotations
import copy
import numpy as np
from simulation.models import Genome, Gene, MetabolicType, DietPreference


def mutate_genome(genome: Genome, mutation_strength: float = 0.05, parent_id: str | None = None, rng: np.random.RandomState | None = None) -> Genome:
    """Create a mutated copy of a genome.

    Args:
        genome: The parent genome to mutate.
        mutation_strength: Fraction of gene range for mutation magnitude.
        parent_id: ID of the parent species to record in lineage.
        rng: Random state for reproducibility.

    Returns:
        A new Genome with mutations applied.
    """
    if rng is None:
        rng = np.random.RandomState()

    new_genes: dict[str, Gene | str] = {}
    for name, gene in genome.genes.items():
        if isinstance(gene, Gene):
            if rng.random() < gene.mutation_rate:
                delta_range = (gene.max_value - gene.min_value) * mutation_strength
                delta = rng.normal(0, delta_range)
                new_val = gene.value + delta
                new_val = max(gene.min_value, min(gene.max_value, new_val))
                new_genes[name] = Gene(
                    value=new_val,
                    min_value=gene.min_value,
                    max_value=gene.max_value,
                    mutation_rate=gene.mutation_rate,
                    dominance=gene.dominance,
                )
            else:
                new_genes[name] = gene.model_copy(deep=True)
        else:
            # Enum gene — small chance to flip
            if rng.random() < 0.01:
                if name == "metabolic_type":
                    options = [e.value for e in MetabolicType]
                    new_genes[name] = rng.choice(options)
                elif name == "diet_preference":
                    options = [e.value for e in DietPreference]
                    new_genes[name] = rng.choice(options)
                else:
                    new_genes[name] = gene
            else:
                new_genes[name] = gene

    new_parent_ids = list(genome.parent_ids)
    if parent_id is not None:
        new_parent_ids.append(parent_id)

    return Genome(
        genes=new_genes,
        generation=genome.generation + 1,
        parent_ids=new_parent_ids,
    )


def crossover_genomes(parent_a: Genome, parent_b: Genome, rng: np.random.RandomState | None = None) -> Genome:
    """Create a child genome by crossing two parents.

    Each gene is randomly selected from one parent, with a chance of blending for float genes.
    """
    if rng is None:
        rng = np.random.RandomState()

    all_keys = set(parent_a.genes.keys()) | set(parent_b.genes.keys())
    child_genes: dict[str, Gene | str] = {}

    for name in all_keys:
        if name not in parent_a.genes:
            child_genes[name] = copy.deepcopy(parent_b.genes[name])
        elif name not in parent_b.genes:
            child_genes[name] = copy.deepcopy(parent_a.genes[name])
        else:
            gene_a = parent_a.genes[name]
            gene_b = parent_b.genes[name]

            if isinstance(gene_a, Gene) and isinstance(gene_b, Gene):
                # Blend: weighted average with noise
                alpha = rng.uniform(0.3, 0.7)
                blended_val = gene_a.value * alpha + gene_b.value * (1 - alpha)
                blended_val += rng.normal(0, (gene_a.max_value - gene_a.min_value) * 0.02)
                blended_val = max(gene_a.min_value, min(gene_a.max_value, blended_val))
                child_genes[name] = Gene(
                    value=blended_val,
                    min_value=gene_a.min_value,
                    max_value=gene_a.max_value,
                    mutation_rate=(gene_a.mutation_rate + gene_b.mutation_rate) / 2,
                    dominance=(gene_a.dominance + gene_b.dominance) / 2,
                )
            else:
                # Enum gene — randomly pick one parent
                child_genes[name] = gene_a if rng.random() < 0.5 else gene_b

    return Genome(
        genes=child_genes,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[],
    )


def calculate_genetic_distance(genome_a: Genome, genome_b: Genome) -> float:
    """Calculate normalized genetic distance between two genomes.

    Returns a value between 0.0 (identical) and ~1.0 (maximally different).
    """
    total_distance = 0.0
    count = 0

    for name in genome_a.genes:
        if name not in genome_b.genes:
            continue
        a = genome_a.genes[name]
        b = genome_b.genes[name]

        if isinstance(a, Gene) and isinstance(b, Gene):
            gene_range = a.max_value - a.min_value
            if gene_range > 0:
                total_distance += abs(a.value - b.value) / gene_range
            count += 1
        else:
            # Enum: 0 if same, 1 if different
            if a != b:
                total_distance += 1.0
            count += 1

    return total_distance / max(count, 1)
```

**3.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_genetics.py -v
```

预期输出: `7 passed`

### Commit
```
git add -A && git commit -m "feat: implement genetics system — mutation, crossover, genetic distance"
```

---

## Task 4: 环境系统 — 温度模型

### 目标
实现温度场计算：纬度梯度 + 季节循环 + 温室效应 + 随机扰动。

### 文件
- `backend/simulation/environment.py` (新建)
- `backend/tests/test_environment.py` (新建)

### 步骤

**4.1 写失败测试**

```python
# backend/tests/test_environment.py
import numpy as np
from simulation.environment import Environment, PlanetConfig

class TestPlanetConfig:
    def test_default_titan_config(self):
        config = PlanetConfig.titan()
        assert config.name == "Titan"
        assert config.gravity < 10.0  # Titan gravity ~1.35 m/s²
        assert config.surface_temp < -100.0

    def test_custom_config(self):
        config = PlanetConfig(
            name="TestPlanet",
            gravity=9.8,
            surface_temp=20.0,
            albedo=0.3,
            axial_tilt=23.5,
            orbital_distance=1.0,
            atmospheric_pressure=1.0,
            co2_ratio=0.0004,
            ch4_ratio=0.0,
        )
        assert config.name == "TestPlanet"

class TestEnvironment:
    def test_create_environment(self, small_grid_size):
        config = PlanetConfig.titan()
        env = Environment(config, size=small_grid_size)
        assert env.temperature.shape == (small_grid_size, small_grid_size)
        assert env.resources.shape == (small_grid_size, small_grid_size)
        assert env.light.shape == (small_grid_size, small_grid_size)

    def test_temperature_has_latitude_gradient(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        env = Environment(config, size=small_grid_size)
        # Equator (middle rows) should be warmer than poles (top/bottom rows)
        equator_temp = np.mean(env.temperature[small_grid_size // 2 - 1 : small_grid_size // 2 + 1, :])
        pole_temp = np.mean(env.temperature[0:2, :])
        assert equator_temp > pole_temp

    def test_greenhouse_effect_warms_planet(self, small_grid_size):
        config_cold = PlanetConfig(surface_temp=20.0, co2_ratio=0.0004, ch4_ratio=0.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0)
        config_warm = PlanetConfig(surface_temp=20.0, co2_ratio=0.15, ch4_ratio=0.05, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0)
        env_cold = Environment(config_cold, size=small_grid_size)
        env_warm = Environment(config_warm, size=small_grid_size)
        assert np.mean(env_warm.temperature) > np.mean(env_cold.temperature)

    def test_seasonal_cycle_changes_temperature(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=23.5, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        env = Environment(config, size=small_grid_size)
        temp_tick_0 = env.temperature.copy()
        env.advance_season(ticks=100)
        temp_tick_100 = env.temperature.copy()
        # Temperature should change due to seasonal cycle
        assert not np.allclose(temp_tick_0, temp_tick_100, atol=0.01)

    def test_resources_have_initial_distribution(self, small_grid_size):
        config = PlanetConfig.titan()
        env = Environment(config, size=small_grid_size)
        assert np.all(env.resources >= 0)
        assert np.mean(env.resources) > 0
```

**4.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_environment.py -v
```

预期输出: `FAILED - ModuleNotFoundError: No module named 'simulation.environment'`

**4.3 实现环境系统**

```python
# backend/simulation/environment.py
from __future__ import annotations
import numpy as np
from pydantic import BaseModel, Field


class PlanetConfig(BaseModel):
    """Physical parameters of a planet."""
    name: str = "Unknown"
    gravity: float = 9.8            # m/s²
    surface_temp: float = 15.0      # °C, base equilibrium temperature
    albedo: float = 0.3             # 0-1, fraction of light reflected
    axial_tilt: float = 23.5        # degrees, drives seasons
    orbital_distance: float = 1.0   # AU
    atmospheric_pressure: float = 1.0  # atm
    co2_ratio: float = 0.0004       # atmospheric CO2 fraction
    ch4_ratio: float = 0.0          # atmospheric CH4 fraction
    magnetic_field: float = 1.0     # relative to Earth
    season_period: int = 365        # ticks per full seasonal cycle

    @staticmethod
    def titan() -> PlanetConfig:
        return PlanetConfig(
            name="Titan",
            gravity=1.35,
            surface_temp=-179.0,
            albedo=0.22,
            axial_tilt=26.7,
            orbital_distance=9.5,
            atmospheric_pressure=1.45,
            co2_ratio=0.0,
            ch4_ratio=0.05,
            magnetic_field=0.0,
            season_period=10759,  # Saturn's orbital period in Earth days
        )

    @staticmethod
    def mars() -> PlanetConfig:
        return PlanetConfig(
            name="Mars",
            gravity=3.72,
            surface_temp=-60.0,
            albedo=0.25,
            axial_tilt=25.2,
            orbital_distance=1.52,
            atmospheric_pressure=0.006,
            co2_ratio=0.95,
            ch4_ratio=0.0,
            magnetic_field=0.0,
            season_period=687,
        )

    @staticmethod
    def europa() -> PlanetConfig:
        return PlanetConfig(
            name="Europa",
            gravity=1.315,
            surface_temp=-160.0,
            albedo=0.67,
            axial_tilt=0.1,
            orbital_distance=5.2,
            atmospheric_pressure=0.0000001,
            co2_ratio=0.0,
            ch4_ratio=0.0,
            magnetic_field=0.0,
            season_period=3.55 * 365,
        )

    @staticmethod
    def kepler442b() -> PlanetConfig:
        return PlanetConfig(
            name="Kepler-442b",
            gravity=12.16,
            surface_temp=-2.0,
            albedo=0.3,
            axial_tilt=15.0,
            orbital_distance=0.409,
            atmospheric_pressure=1.5,
            co2_ratio=0.01,
            ch4_ratio=0.001,
            magnetic_field=1.2,
            season_period=112,
        )


class Environment:
    """Manages the planet's environmental layers: temperature, resources, light, water."""

    def __init__(self, config: PlanetConfig, size: int = 50, rng: np.random.RandomState | None = None):
        self.config = config
        self.size = size
        self.rng = rng if rng is not None else np.random.RandomState()
        self.tick_count = 0

        # Build latitude array: 0 = north pole, size-1 = south pole
        self.latitudes = np.linspace(90, -90, size)  # degrees

        # Initialize layers
        self.temperature = self._init_temperature()
        self.resources = self._init_resources()
        self.light = self._init_light()
        self.water = self._init_water()

        # Atmosphere composition (global scalars)
        self.atmosphere = {
            "O2": 0.0,
            "CO2": config.co2_ratio,
            "CH4": config.ch4_ratio,
            "N2": 1.0 - config.co2_ratio - config.ch4_ratio,
            "Pressure": config.atmospheric_pressure,
        }

        # Volcanic hotspots
        self.volcanic_heat = np.zeros((size, size))
        self._init_volcanoes(n_volcanoes=max(1, size // 10))

    def _init_temperature(self) -> np.ndarray:
        """Temperature field with latitude gradient."""
        # Base temp from planet config
        base = self.config.surface_temp

        # Latitude gradient: poles colder, equator warmer
        # Scale by distance (closer to star = more variation)
        gradient_strength = 30.0 / max(self.config.orbital_distance, 0.1)
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)  # 1 at equator, 0 at poles
        gradient = np.outer(lat_factor, np.ones(self.size)) * gradient_strength

        # Greenhouse warming
        greenhouse = self._greenhouse_effect()

        temp = base + gradient + greenhouse
        # Add small noise
        temp += self.rng.normal(0, 1.0, (self.size, self.size))
        return temp

    def _greenhouse_effect(self) -> float:
        """Calculate greenhouse warming from CO2 and CH4."""
        # Simplified logarithmic greenhouse model
        co2_warming = 3.0 * np.log(1 + self.config.co2_ratio / 0.0004) if self.config.co2_ratio > 0 else 0.0
        ch4_warming = 8.0 * np.log(1 + self.config.ch4_ratio / 0.0001) if self.config.ch4_ratio > 0 else 0.0
        return co2_warming + ch4_warming

    def _init_resources(self) -> np.ndarray:
        """Resource distribution — richer near center, some noise."""
        resources = np.ones((self.size, self.size)) * 0.5
        # Center enrichment
        center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        resources += 0.5 * np.exp(-dist ** 2 / (2 * (self.size / 3) ** 2))
        # Noise
        resources += self.rng.normal(0, 0.1, (self.size, self.size))
        return np.clip(resources, 0.1, 2.0)

    def _init_light(self) -> np.ndarray:
        """Light intensity — depends on distance from star and latitude."""
        base_light = 1.0 / (self.config.orbital_distance ** 2)
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)
        light = base_light * np.outer(lat_factor, np.ones(self.size))
        # Attennuate by atmosphere (pressure affects scattering)
        light *= min(1.0, self.config.atmospheric_pressure)
        return np.clip(light, 0.0, 2.0)

    def _init_water(self) -> np.ndarray:
        """Water/liquid availability — temperature dependent."""
        # Water exists where temperature allows liquid phase
        # For simplicity: more water near equator, less at poles
        lat_rad = np.deg2rad(self.latitudes)
        lat_factor = np.cos(lat_rad)
        water = 0.3 + 0.4 * np.outer(lat_factor, np.ones(self.size))
        water += self.rng.normal(0, 0.05, (self.size, self.size))
        return np.clip(water, 0.0, 1.0)

    def _init_volcanoes(self, n_volcanoes: int):
        """Place random volcanic hotspots."""
        for _ in range(n_volcanoes):
            vy = self.rng.randint(0, self.size)
            vx = self.rng.randint(0, self.size)
            radius = self.rng.randint(2, max(3, self.size // 5))
            y, x = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
            self.volcanic_heat += 20.0 * np.exp(-dist ** 2 / (2 * radius ** 2))

    def advance_season(self, ticks: int = 1):
        """Advance seasonal cycle by given ticks."""
        self.tick_count += ticks

    def update(self, ticks: int = 1, biomass_heat: np.ndarray | None = None):
        """Update environment for one or more simulation ticks.

        Args:
            ticks: Number of ticks to advance.
            biomass_heat: Optional 2D array of heat generated by biological activity.
        """
        self.tick_count += ticks
        t = self.tick_count

        # 1. Seasonal temperature oscillation
        season_phase = 2 * np.pi * t / max(self.config.season_period, 1)
        seasonal_amplitude = self.config.axial_tilt * 0.5  # degrees → °C proxy
        lat_rad = np.deg2rad(self.latitudes)
        # Northern hemisphere summer = southern hemisphere winter
        seasonal_offset = seasonal_amplitude * np.sin(season_phase) * np.sin(lat_rad)
        seasonal_grid = np.outer(seasonal_offset, np.ones(self.size))

        # 2. Greenhouse (may change if atmosphere changes)
        greenhouse = self._greenhouse_effect()

        # 3. Volcanic heat (decays slowly)
        self.volcanic_heat *= 0.999

        # 4. Biological heat
        bio_heat = biomass_heat if biomass_heat is not None else np.zeros((self.size, self.size))

        # 5. Random perturbation
        noise = self.rng.normal(0, 0.3, (self.size, self.size))

        # 6. Compute final temperature
        base = self.config.surface_temp
        lat_gradient = 30.0 / max(self.config.orbital_distance, 0.1)
        lat_factor = np.cos(lat_rad)
        gradient = np.outer(lat_factor, np.ones(self.size)) * lat_gradient

        self.temperature = base + gradient + greenhouse + seasonal_grid + self.volcanic_heat + bio_heat + noise

        # 7. Resource regeneration
        # Light promotes resource regeneration
        regen_rate = 0.01 * self.light
        self.resources = np.clip(self.resources + regen_rate, 0.0, 2.0)

    def get_snapshot(self) -> dict:
        """Return a serializable snapshot of the environment state."""
        return {
            "tick": self.tick_count,
            "temperature_mean": float(np.mean(self.temperature)),
            "temperature_min": float(np.min(self.temperature)),
            "temperature_max": float(np.max(self.temperature)),
            "resources_mean": float(np.mean(self.resources)),
            "light_mean": float(np.mean(self.light)),
            "atmosphere": dict(self.atmosphere),
        }
```

**4.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_environment.py -v
```

预期输出: `6 passed`

### Commit
```
git add -A && git commit -m "feat: implement environment system — temperature, resources, seasons, volcanoes"
```

---

## Task 5: 物种模型

### 目标
将基因组 + 空间分布组合为完整的物种实体。

### 文件
- `backend/simulation/species.py` (新建)
- `backend/tests/test_species.py` (新建)

### 步骤

**5.1 写失败测试**

```python
# backend/tests/test_species.py
import numpy as np
from simulation.models import Genome
from simulation.species import Species

class TestSpecies:
    def test_create_species(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.id == "sp_001"
        assert sp.name == "TestPlant"
        assert sp.biomass.shape == (small_grid_size, small_grid_size)
        assert np.sum(sp.biomass) > 0

    def test_species_total_biomass(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.total_biomass() > 0

    def test_species_is_alive(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        assert sp.is_alive()

    def test_species_dies_when_biomass_zero(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size)
        sp.biomass = np.zeros((small_grid_size, small_grid_size))
        assert not sp.is_alive()

    def test_species_center_seed(self, small_grid_size):
        genome = Genome.create_default()
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size, seed_area="center")
        mid = small_grid_size // 2
        assert sp.biomass[mid, mid] > 0
        assert sp.biomass[0, 0] == 0.0

    def test_species_random_seed(self, small_grid_size):
        genome = Genome.create_default()
        rng = np.random.RandomState(42)
        sp = Species.create("sp_001", "TestPlant", genome, grid_size=small_grid_size, seed_area="random", rng=rng)
        assert sp.total_biomass() > 0
```

**5.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_species.py -v
```

**5.3 实现物种模型**

```python
# backend/simulation/species.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from simulation.models import Genome, MetabolicType


@dataclass
class EvolutionEvent:
    tick: int
    event_type: str  # "mutation", "speciation", "extinction", "adaptation"
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class Species:
    id: str
    name: str
    genome: Genome
    biomass: np.ndarray
    ancestor_id: str | None = None
    extinction_tick: int | None = None
    color: tuple[int, int, int] = (100, 200, 100)
    history: list[EvolutionEvent] = field(default_factory=list)

    @staticmethod
    def create(
        species_id: str,
        name: str,
        genome: Genome,
        grid_size: int = 50,
        initial_biomass: float = 0.5,
        seed_area: str = "center",
        rng: np.random.RandomState | None = None,
    ) -> Species:
        """Create a new species with spatial biomass distribution.

        Args:
            species_id: Unique identifier.
            name: Display name.
            genome: The species' genetic code.
            grid_size: Size of the spatial grid.
            initial_biomass: Starting biomass concentration.
            seed_area: Where to seed — "center", "random", or "everywhere".
            rng: Random state for reproducibility.

        Returns:
            A new Species instance.
        """
        if rng is None:
            rng = np.random.RandomState()

        biomass = np.zeros((grid_size, grid_size))

        if seed_area == "center":
            mid = grid_size // 2
            radius = max(2, grid_size // 8)
            y, x = np.ogrid[:grid_size, :grid_size]
            dist = np.sqrt((x - mid) ** 2 + (y - mid) ** 2)
            mask = dist < radius
            biomass[mask] = initial_biomass
        elif seed_area == "random":
            n_patches = rng.randint(3, 8)
            for _ in range(n_patches):
                py, px = rng.randint(0, grid_size, 2)
                radius = rng.randint(1, max(2, grid_size // 6))
                y, x = np.ogrid[:grid_size, :grid_size]
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                mask = dist < radius
                biomass[mask] = initial_biomass * rng.uniform(0.3, 1.0)
        elif seed_area == "everywhere":
            biomass[:] = initial_biomass * 0.1
            biomass += rng.uniform(0, 0.05, (grid_size, grid_size))

        # Assign color based on metabolic type
        meta_type = genome.get_enum("metabolic_type")
        if meta_type == MetabolicType.PHOTOSYNTHESIS.value:
            color = (50, 180, 50)  # Green
        elif meta_type == MetabolicType.CHEMOSYNTHESIS.value:
            color = (50, 100, 200)  # Blue
        else:
            color = (200, 80, 80)  # Red

        return Species(
            id=species_id,
            name=name,
            genome=genome,
            biomass=biomass,
            color=color,
        )

    def total_biomass(self) -> float:
        return float(np.sum(self.biomass))

    def is_alive(self) -> bool:
        return self.total_biomass() > 0.001

    def mean_fitness(self) -> float:
        """Placeholder — will be computed by ecosystem using environment data."""
        return 0.5
```

**5.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_species.py -v
```

预期输出: `6 passed`

### Commit
```
git add -A && git commit -m "feat: implement Species model with spatial biomass distribution"
```

---

## Task 6: 食物网系统

### 目标
实现基于基因的涌现式食物网：生产者/消费者/捕食者/分解者的能量流动。

### 文件
- `backend/simulation/food_web.py` (新建)
- `backend/tests/test_food_web.py` (新建)

### 步骤

**6.1 写失败测试**

```python
# backend/tests/test_food_web.py
import numpy as np
from simulation.models import Genome, Gene, MetabolicType, DietPreference
from simulation.species import Species
from simulation.food_web import FoodWeb

class TestFoodWeb:
    def test_producer_is_not_prey(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        # Producer should not be prey for producer
        assert producer.id not in web.get_prey(producer.id)

    def test_consumer_eats_producer(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        prey_list = web.get_prey(consumer.id)
        assert producer.id in prey_list

    def test_omni_eats_both(self, small_grid_size):
        web = FoodWeb()
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        consumer = Species.create("c1", "Herbivore", Genome.create_consumer(), grid_size=small_grid_size)
        omni_genome = Genome.create_consumer()
        omni_genome.genes["diet_preference"] = DietPreference.OMNI.value
        omni = Species.create("o1", "Omnivore", omni_genome, grid_size=small_grid_size)
        web.register(producer)
        web.register(consumer)
        web.register(omni)
        prey_list = web.get_prey(omni.id)
        assert producer.id in prey_list
        assert consumer.id in prey_list

    def test_attack_power_formula(self):
        web = FoodWeb()
        # attack = body_size * mobility * sensory_range
        genome = Genome.create_consumer()
        genome.genes["body_size"] = Gene(value=3.0, min_value=0.1, max_value=10.0)
        genome.genes["mobility"] = Gene(value=0.5, min_value=0.0, max_value=1.0)
        genome.genes["sensory_range"] = Gene(value=4.0, min_value=1.0, max_value=15.0)
        attack = web._calc_attack_power(genome)
        assert abs(attack - 3.0 * 0.5 * 4.0) < 0.001

    def test_predation_efficiency(self):
        web = FoodWeb()
        # efficiency = attack / (attack + defense)
        eff = web.predation_efficiency(attack_power=5.0, prey_defense=5.0)
        assert abs(eff - 0.5) < 0.001

    def test_predation_efficiency_no_defense(self):
        web = FoodWeb()
        eff = web.predation_efficiency(attack_power=5.0, prey_defense=0.0)
        assert abs(eff - 1.0) < 0.001
```

**6.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_food_web.py -v
```

**6.3 实现食物网**

```python
# backend/simulation/food_web.py
from __future__ import annotations
import numpy as np
from simulation.models import MetabolicType, DietPreference, Genome
from simulation.species import Species


class FoodWeb:
    """Emergent food web based on species genetics.

    Food relationships are not hardcoded — they emerge from metabolic_type
    and diet_preference genes.
    """

    def __init__(self):
        self.species_map: dict[str, Species] = {}

    def register(self, species: Species):
        self.species_map[species.id] = species

    def unregister(self, species_id: str):
        self.species_map.pop(species_id, None)

    def get_prey(self, predator_id: str) -> list[str]:
        """Get list of species IDs that the given predator can eat."""
        predator = self.species_map.get(predator_id)
        if predator is None:
            return []

        meta_type = predator.genome.get_enum("metabolic_type")
        diet = predator.genome.get_enum("diet_preference")

        # Producers don't eat anything
        if meta_type == MetabolicType.PHOTOSYNTHESIS.value or meta_type == MetabolicType.CHEMOSYNTHESIS.value:
            return []

        prey_ids = []
        for sid, sp in self.species_map.items():
            if sid == predator_id:
                continue
            if not sp.is_alive():
                continue

            sp_meta = sp.genome.get_enum("metabolic_type")

            if diet == DietPreference.PRODUCER.value:
                # This shouldn't happen for heterotrophs, but handle gracefully
                pass
            elif diet == DietPreference.CONSUMER.value:
                # Eats producers (photosynthesis or chemosynthesis)
                if sp_meta in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value):
                    prey_ids.append(sid)
            elif diet == DietPreference.OMNI.value:
                # Eats everyone (less efficiently)
                prey_ids.append(sid)

        return prey_ids

    def _calc_attack_power(self, genome: Genome) -> float:
        """Calculate attack power from genetics: body_size * mobility * sensory_range."""
        body = genome.get_float("body_size")
        mob = genome.get_float("mobility")
        sense = genome.get_float("sensory_range")
        return body * mob * sense

    @staticmethod
    def predation_efficiency(attack_power: float, prey_defense: float) -> float:
        """Predation success rate: attack / (attack + defense)."""
        if attack_power + prey_defense == 0:
            return 0.0
        return attack_power / (attack_power + prey_defense)

    def compute_predation_rates(self) -> dict[tuple[str, str], float]:
        """Compute predation rate matrix: (predator_id, prey_id) → rate.

        Returns dict mapping (predator_id, prey_id) to predation efficiency.
        """
        rates = {}
        for pred_id, predator in self.species_map.items():
            if not predator.is_alive():
                continue
            prey_ids = self.get_prey(pred_id)
            attack = self._calc_attack_power(predator.genome)
            for prey_id in prey_ids:
                prey = self.species_map[prey_id]
                defense = prey.genome.get_float("defense")
                eff = self.predation_efficiency(attack, defense)
                rates[(pred_id, prey_id)] = eff
        return rates

    def get_producers(self) -> list[str]:
        """Get IDs of all producer species."""
        return [
            sid for sid, sp in self.species_map.items()
            if sp.genome.get_enum("metabolic_type") in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value)
            and sp.is_alive()
        ]

    def get_consumers(self) -> list[str]:
        """Get IDs of all consumer (heterotroph) species."""
        return [
            sid for sid, sp in self.species_map.items()
            if sp.genome.get_enum("metabolic_type") == MetabolicType.HETEROTROPHY.value
            and sp.is_alive()
        ]

    def get_trophic_level(self, species_id: str) -> float:
        """Estimate trophic level: 1.0 for producers, 2.0 for primary consumers, etc."""
        sp = self.species_map.get(species_id)
        if sp is None:
            return 0.0
        meta = sp.genome.get_enum("metabolic_type")
        if meta in (MetabolicType.PHOTOSYNTHESIS.value, MetabolicType.CHEMOSYNTHESIS.value):
            return 1.0
        diet = sp.genome.get_enum("diet_preference")
        if diet == DietPreference.CONSUMER.value:
            return 2.0
        if diet == DietPreference.OMNI.value:
            return 2.5
        return 2.0
```

**6.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_food_web.py -v
```

预期输出: `6 passed`

### Commit
```
git add -A && git commit -m "feat: implement emergent food web based on species genetics"
```

---

## Task 7: 模拟引擎 — 主循环

### 目标
将所有组件组合为完整的模拟引擎：环境更新 → 适应度计算 → 物种动态 → 环境反馈。

### 文件
- `backend/simulation/engine.py` (新建)
- `backend/tests/test_engine.py` (新建)

### 步骤

**7.1 写失败测试**

```python
# backend/tests/test_engine.py
import numpy as np
from simulation.engine import SimulationEngine
from simulation.models import Genome
from simulation.species import Species
from simulation.environment import PlanetConfig

class TestSimulationEngine:
    def test_create_engine(self, small_grid_size):
        config = PlanetConfig.titan()
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        assert engine.tick == 0
        assert len(engine.species_list) == 0

    def test_add_species(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        assert len(engine.species_list) == 1

    def test_engine_step_advances_tick(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        engine.step()
        assert engine.tick == 1

    def test_producer_grows_with_resources(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size, initial_biomass=0.1)
        initial_total = producer.total_biomass()
        engine.add_species(producer)
        # Run a few steps — producer should grow if temperature is reasonable
        for _ in range(10):
            engine.step()
        assert producer.total_biomass() > initial_total * 0.5  # At least not dead

    def test_species_extinction_removes_from_list(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        # Force extinction
        producer.biomass = np.zeros((small_grid_size, small_grid_size))
        engine.step()
        # Species should be removed or marked dead
        alive = [s for s in engine.species_list if s.is_alive()]
        assert len(alive) == 0

    def test_get_snapshot(self, small_grid_size):
        config = PlanetConfig(surface_temp=20.0, axial_tilt=0.0, albedo=0.3, orbital_distance=1.0, gravity=9.8, atmospheric_pressure=1.0, co2_ratio=0.0004, ch4_ratio=0.0)
        engine = SimulationEngine(config=config, grid_size=small_grid_size)
        producer = Species.create("p1", "Plant", Genome.create_default(), grid_size=small_grid_size)
        engine.add_species(producer)
        engine.step()
        snapshot = engine.get_snapshot()
        assert "tick" in snapshot
        assert "species" in snapshot
        assert "environment" in snapshot
        assert len(snapshot["species"]) == 1
```

**7.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_engine.py -v
```

**7.3 实现模拟引擎**

```python
# backend/simulation/engine.py
from __future__ import annotations
import numpy as np
from scipy.signal import convolve2d
from simulation.environment import Environment, PlanetConfig
from simulation.species import Species, EvolutionEvent
from simulation.food_web import FoodWeb


class SimulationEngine:
    """Core simulation loop: environment ↔ species interactions."""

    def __init__(self, config: PlanetConfig, grid_size: int = 50, rng: np.random.RandomState | None = None):
        self.config = config
        self.grid_size = grid_size
        self.rng = rng if rng is not None else np.random.RandomState()
        self.env = Environment(config, size=grid_size, rng=self.rng)
        self.food_web = FoodWeb()
        self.species_list: list[Species] = []
        self.tick = 0
        self.events: list[EvolutionEvent] = []

        # Diffusion kernel for biomass spread
        self.diffusion_kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05],
        ])

    def add_species(self, species: Species):
        self.species_list.append(species)
        self.food_web.register(species)

    def step(self):
        """Execute one simulation tick."""
        self.tick += 1

        # 1. Compute biomass-generated heat
        biomass_heat = np.zeros((self.grid_size, self.grid_size))
        for sp in self.species_list:
            if sp.is_alive():
                biomass_heat += sp.biomass * 0.01

        # 2. Update environment
        self.env.update(ticks=1, biomass_heat=biomass_heat)

        # 3. Process each species
        dead_species = []
        for sp in self.species_list:
            if not sp.is_alive():
                dead_species.append(sp)
                continue
            self._process_species(sp)

        # 4. Remove dead species
        for sp in dead_species:
            sp.extinction_tick = self.tick
            sp.history.append(EvolutionEvent(
                tick=self.tick,
                event_type="extinction",
                description=f"{sp.name} went extinct at tick {self.tick}",
            ))
            self.events.append(sp.history[-1])
            self.food_web.unregister(sp.id)

        self.species_list = [s for s in self.species_list if s.is_alive()]

        # 5. Resource regeneration (producers add resources back via decomposition)
        # Handled in environment.update

    def _process_species(self, sp: Species):
        """Process births, deaths, movement, and predation for one species."""
        meta_type = sp.genome.get_enum("metabolic_type")
        body_size = sp.genome.get_float("body_size")
        rep_rate = sp.genome.get_float("reproduction_rate")
        rep_cost = sp.genome.get_float("reproduction_cost")
        mobility = sp.genome.get_float("mobility")
        lifespan = sp.genome.get_float("lifespan")
        temp_opt = sp.genome.get_float("temp_optimum")
        temp_tol = sp.genome.get_float("temp_tolerance")

        # A. Temperature fitness (Gaussian)
        temp_dist = np.abs(self.env.temperature - temp_opt)
        temp_fitness = np.exp(-(temp_dist ** 2) / (2 * temp_tol ** 2))

        # B. Resource fitness
        if meta_type in ("photosynthesis", "chemosynthesis"):
            # Producers use light + resources
            resource_fitness = np.clip(self.env.light * 2.0, 0, 2) * np.clip(self.env.resources, 0, 2)
        else:
            # Consumers use resources (representing prey availability)
            resource_fitness = np.clip(self.env.resources, 0, 2)

        # C. Competition pressure
        total_biomass_here = np.zeros((self.grid_size, self.grid_size))
        for other in self.species_list:
            if other.id != sp.id and other.is_alive():
                total_biomass_here += other.biomass
        competition = 1.0 / (1.0 + total_biomass_here * 0.1)

        # D. Combined fitness
        fitness = temp_fitness * resource_fitness * competition
        fitness = np.clip(fitness, 0, 3)

        # E. Births
        births = sp.biomass * rep_rate * fitness * 0.1
        # Cost: births consume resources
        resource_cost = births * rep_cost * body_size * 0.1
        self.env.resources = np.clip(self.env.resources - resource_cost, 0, 2)

        # F. Deaths
        # Base mortality (age)
        base_mortality = 1.0 / max(lifespan, 1)
        # Environmental stress (temperature out of range)
        env_stress = np.clip(1.0 - temp_fitness, 0, 1) * 0.05
        # Starvation (low resources)
        starvation = np.clip(1.0 - resource_fitness, 0, 1) * 0.02
        deaths = sp.biomass * (base_mortality + env_stress + starvation)

        # G. Predation
        predation_rates = self.food_web.compute_predation_rates()
        for (pred_id, prey_id), rate in predation_rates.items():
            if prey_id == sp.id:
                predator = None
                for s in self.species_list:
                    if s.id == pred_id:
                        predator = s
                        break
                if predator is not None:
                    # Predator consumes prey biomass
                    consumed = sp.biomass * rate * predator.biomass * 0.05
                    consumed = np.minimum(consumed, sp.biomass)
                    sp.biomass -= consumed
                    # Predator gains energy
                    predator.biomass += consumed * 0.5  # energy transfer efficiency

        # H. Update biomass
        sp.biomass = sp.biomass + births - deaths
        sp.biomass = np.clip(sp.biomass, 0, 50)

        # I. Diffusion (movement)
        if mobility > 0.01:
            diffusion_strength = mobility * 0.3
            diffused = convolve2d(sp.biomass, self.diffusion_kernel, mode='same')
            sp.biomass = sp.biomass * (1 - diffusion_strength) + diffused * diffusion_strength

        # J. Producers: consume CO2, produce O2
        if meta_type == "photosynthesis":
            o2_production = np.sum(sp.biomass) * 0.0001
            co2_consumption = np.sum(sp.biomass) * 0.0001
            self.env.atmosphere["O2"] = min(0.5, self.env.atmosphere.get("O2", 0) + o2_production)
            self.env.atmosphere["CO2"] = max(0, self.env.atmosphere.get("CO2", 0) - co2_consumption)

        # K. Consumers: consume O2, produce CO2
        if meta_type == "heterotrophy":
            o2_consumption = np.sum(sp.biomass) * 0.00005
            co2_production = np.sum(sp.biomass) * 0.00005
            self.env.atmosphere["O2"] = max(0, self.env.atmosphere.get("O2", 0) - o2_consumption)
            self.env.atmosphere["CO2"] = min(1.0, self.env.atmosphere.get("CO2", 0) + co2_production)

    def get_snapshot(self) -> dict:
        """Return a serializable snapshot of the full simulation state."""
        return {
            "tick": self.tick,
            "environment": self.env.get_snapshot(),
            "species": [
                {
                    "id": sp.id,
                    "name": sp.name,
                    "total_biomass": sp.total_biomass(),
                    "is_alive": sp.is_alive(),
                    "metabolic_type": sp.genome.get_enum("metabolic_type"),
                    "color": sp.color,
                }
                for sp in self.species_list
            ],
            "events": [
                {"tick": e.tick, "type": e.event_type, "description": e.description}
                for e in self.events[-10:]  # Last 10 events
            ],
        }
```

**7.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_engine.py -v
```

预期输出: `6 passed`

### Commit
```
git add -A && git commit -m "feat: implement simulation engine with species dynamics and food web integration"
```

---

## Task 8: 预设星球与物种模板 (JSON 配置)

### 目标
创建可加载的星球和物种模板 JSON 文件。

### 文件
- `backend/config/planets/titan.json` (新建)
- `backend/config/planets/mars.json` (新建)
- `backend/config/planets/europa.json` (新建)
- `backend/config/planets/kepler442b.json` (新建)
- `backend/config/species/producer_photo.json` (新建)
- `backend/config/species/consumer_herbivore.json` (新建)
- `backend/config/species/producer_chemo.json` (新建)
- `backend/simulation/templates.py` (新建)
- `backend/tests/test_templates.py` (新建)

### 步骤

**8.1 写失败测试**

```python
# backend/tests/test_templates.py
from simulation.templates import load_planet_config, load_species_template, list_planets

class TestTemplates:
    def test_list_planets(self):
        planets = list_planets()
        assert "titan" in planets
        assert "mars" in planets
        assert "europa" in planets

    def test_load_titan(self):
        config = load_planet_config("titan")
        assert config.name == "Titan"
        assert config.surface_temp < -100

    def test_load_species_template(self):
        genome = load_species_template("producer_photo")
        assert genome.get_enum("metabolic_type") == "photosynthesis"
```

**8.2 运行测试确认失败**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_templates.py -v
```

**8.3 创建配置文件和加载逻辑**

创建所有 JSON 配置文件（星球模板和物种模板），以及 `templates.py` 加载模块。具体代码较长，将按模板结构编写。

星球 JSON 格式:
```json
{
  "name": "Titan",
  "gravity": 1.35,
  "surface_temp": -179.0,
  "albedo": 0.22,
  "axial_tilt": 26.7,
  "orbital_distance": 9.5,
  "atmospheric_pressure": 1.45,
  "co2_ratio": 0.0,
  "ch4_ratio": 0.05,
  "magnetic_field": 0.0,
  "season_period": 10759
}
```

物种模板 JSON 格式:
```json
{
  "name": "Photo-Producer",
  "metabolic_type": "photosynthesis",
  "body_size": 0.5,
  "temp_optimum": 25.0,
  "temp_tolerance": 40.0,
  "reproduction_rate": 1.0,
  "defense": 0.05,
  "mobility": 0.0,
  "sensory_range": 1.0
}
```

`templates.py` 提供 `load_planet_config(name)` → `PlanetConfig`，`load_species_template(name)` → `Genome`，`list_planets()` → `list[str]`。

**8.4 运行测试确认通过**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/test_templates.py -v
```

预期输出: `3 passed`

### Commit
```
git add -A && git commit -m "feat: add planet and species template configs with JSON loader"
```

---

## Task 9: Matplotlib 命令行可视化

### 目标
创建命令行运行入口，使用 matplotlib 实时显示模拟状态（延续现有代码风格）。

### 文件
- `backend/visualize.py` (新建)
- `backend/run_sim.py` (新建)

### 步骤

**9.1 实现可视化模块**

```python
# backend/visualize.py
import matplotlib.pyplot as plt
import numpy as np
from simulation.engine import SimulationEngine
from simulation.species import Species
from simulation.environment import PlanetConfig
from simulation.models import Genome
from simulation.templates import load_planet_config, load_species_template


def run_visual_simulation(
    planet_name: str = "titan",
    n_producers: int = 1,
    n_consumers: int = 1,
    steps: int = 2000,
    ai_interval: int = 0,  # 0 = no AI
    grid_size: int = 50,
):
    """Run simulation with matplotlib visualization."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    config = load_planet_config(planet_name)
    engine = SimulationEngine(config=config, grid_size=grid_size)

    # Add producer species
    for i in range(n_producers):
        genome = load_species_template("producer_photo")
        sp = Species.create(f"producer_{i}", f"Producer-{i}", genome, grid_size=grid_size)
        engine.add_species(sp)

    # Add consumer species
    for i in range(n_consumers):
        genome = load_species_template("consumer_herbivore")
        sp = Species.create(f"consumer_{i}", f"Consumer-{i}", genome, grid_size=grid_size, seed_area="random")
        engine.add_species(sp)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Alien Evolution Simulator — {config.name}", fontsize=14)

    pop_history = {sp.id: [] for sp in engine.species_list}
    tick_history = []

    try:
        for step_i in range(steps):
            engine.step()

            # Record history
            tick_history.append(engine.tick)
            for sp in engine.species_list:
                if sp.id in pop_history:
                    pop_history[sp.id].append(sp.total_biomass())

            # Update display every 10 steps
            if step_i % 10 == 0:
                # Panel 1: Combined biomass heatmap
                axes[0, 0].clear()
                total = np.zeros((grid_size, grid_size))
                for sp in engine.species_list:
                    total += sp.biomass
                axes[0, 0].imshow(total, cmap='YlGn', vmin=0, vmax=5)
                axes[0, 0].set_title(f"Biomass (tick {engine.tick})")

                # Panel 2: Temperature
                axes[0, 1].clear()
                axes[0, 1].imshow(engine.env.temperature, cmap='RdYlBu_r')
                axes[0, 1].set_title(f"Temperature (avg {np.mean(engine.env.temperature):.1f}°C)")

                # Panel 3: Resources
                axes[1, 0].clear()
                axes[1, 0].imshow(engine.env.resources, cmap='Greens', vmin=0, vmax=2)
                axes[1, 0].set_title("Resources")

                # Panel 4: Population over time
                axes[1, 1].clear()
                for sp_id, history in pop_history.items():
                    if len(history) > 0:
                        axes[1, 1].plot(tick_history[:len(history)], history, label=sp_id)
                axes[1, 1].set_title("Population")
                axes[1, 1].set_xlabel("Tick")
                axes[1, 1].set_ylabel("Total Biomass")
                axes[1, 1].legend(fontsize=8)

                plt.tight_layout()
                plt.pause(0.01)

            if step_i % 100 == 0:
                alive = [s for s in engine.species_list if s.is_alive()]
                print(f"Tick {engine.tick}: {len(alive)} species alive, "
                      f"env temp={np.mean(engine.env.temperature):.1f}°C, "
                      f"O2={engine.env.atmosphere.get('O2', 0):.4f}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        plt.ioff()
        plt.show(block=True)
```

**9.2 创建运行入口**

```python
# backend/run_sim.py
"""Entry point for running the simulation from command line."""
from visualize import run_visual_simulation

if __name__ == "__main__":
    run_visual_simulation(
        planet_name="titan",
        n_producers=2,
        n_consumers=1,
        steps=2000,
        grid_size=50,
    )
```

**9.3 运行验证**

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python run_sim.py
```

预期输出: matplotlib 窗口弹出，显示 4 面板实时模拟。终端每 100 tick 打印状态。按 Ctrl+C 停止。

### Commit
```
git add -A && git commit -m "feat: add matplotlib visualization and CLI entry point"
```

---

## Task 10: 全套测试套件运行

### 目标
确保所有测试通过，验证模块间集成。

### 步骤

```bash
cd D:\evolution\LifeEvolutionSimulation\backend
python -m pytest tests/ -v --tb=short
```

预期输出:
```
tests/test_models.py::TestGene::test_create_float_gene PASSED
tests/test_models.py::TestGene::test_gene_clamps_value PASSED
tests/test_models.py::TestGene::test_gene_clamps_negative PASSED
tests/test_models.py::TestGenome::test_create_genome_with_defaults PASSED
tests/test_models.py::TestGenome::test_genome_has_all_required_genes PASSED
tests/test_models.py::TestGenome::test_metabolic_type_is_enum PASSED
tests/test_models.py::TestGenome::test_diet_preference_is_enum PASSED
tests/test_genetics.py::TestMutation::test_mutation_changes_value PASSED
tests/test_genetics.py::TestMutation::test_mutation_respects_bounds PASSED
tests/test_genetics.py::TestMutation::test_mutation_increments_generation PASSED
tests/test_genetics.py::TestMutation::test_mutation_preserves_parent_id PASSED
tests/test_genetics.py::TestCrossover::test_crossover_produces_valid_genome PASSED
tests/test_genetics.py::TestCrossover::test_crossover_child_is_different_from_parents PASSED
tests/test_genetics.py::TestGeneticDistance::test_identical_genomes_have_zero_distance PASSED
tests/test_genetics.py::TestGeneticDistance::test_different_genomes_have_positive_distance PASSED
tests/test_environment.py::TestPlanetConfig::test_default_titan_config PASSED
tests/test_environment.py::TestPlanetConfig::test_custom_config PASSED
tests/test_environment.py::TestEnvironment::test_create_environment PASSED
tests/test_environment.py::TestEnvironment::test_temperature_has_latitude_gradient PASSED
tests/test_environment.py::TestEnvironment::test_greenhouse_effect_warms_planet PASSED
tests/test_environment.py::TestEnvironment::test_seasonal_cycle_changes_temperature PASSED
tests/test_environment.py::TestEnvironment::test_resources_have_initial_distribution PASSED
tests/test_species.py::TestSpecies::test_create_species PASSED
tests/test_species.py::TestSpecies::test_species_total_biomass PASSED
tests/test_species.py::TestSpecies::test_species_is_alive PASSED
tests/test_species.py::TestSpecies::test_species_dies_when_biomass_zero PASSED
tests/test_species.py::TestSpecies::test_species_center_seed PASSED
tests/test_species.py::TestSpecies::test_species_random_seed PASSED
tests/test_food_web.py::TestFoodWeb::test_producer_is_not_prey PASSED
tests/test_food_web.py::TestFoodWeb::test_consumer_eats_producer PASSED
tests/test_food_web.py::TestFoodWeb::test_omni_eats_both PASSED
tests/test_food_web.py::TestFoodWeb::test_attack_power_formula PASSED
tests/test_food_web.py::TestFoodWeb::test_predation_efficiency PASSED
tests/test_food_web.py::TestFoodWeb::test_predation_efficiency_no_defense PASSED
tests/test_engine.py::TestSimulationEngine::test_create_engine PASSED
tests/test_engine.py::TestSimulationEngine::test_add_species PASSED
tests/test_engine.py::TestSimulationEngine::test_engine_step_advances_tick PASSED
tests/test_engine.py::TestSimulationEngine::test_producer_grows_with_resources PASSED
tests/test_engine.py::TestSimulationEngine::test_species_extinction_removes_from_list PASSED
tests/test_engine.py::TestSimulationEngine::test_get_snapshot PASSED
tests/test_templates.py::TestTemplates::test_list_planets PASSED
tests/test_templates.py::TestTemplates::test_load_titan PASSED
tests/test_templates.py::TestTemplates::test_load_species_template PASSED

43 passed
```

### Commit
```
git add -A && git commit -m "test: all 43 tests passing — Phase 1 core engine complete"
```

---

## Phase 1 任务总结

| Task | 模块 | 测试数 | 文件数 |
|------|------|--------|--------|
| 1 | 项目结构 | 0 | 4 |
| 2 | 数据模型 | 7 | 3 |
| 3 | 遗传系统 | 7 | 2 |
| 4 | 环境系统 | 6 | 2 |
| 5 | 物种模型 | 6 | 2 |
| 6 | 食物网 | 6 | 2 |
| 7 | 模拟引擎 | 6 | 2 |
| 8 | 配置模板 | 3 | 10+ |
| 9 | 可视化 | 0 | 2 |
| 10 | 集成验证 | 0 | 0 |
| **总计** | | **41** | **~30** |

### Phase 1 完成后可运行效果
- `python run_sim.py` 启动 Titan 环境模拟
- 2 种光合生产者 + 1 种异养消费者在 50x50 网格上协同演化
- matplotlib 实时显示生物量热力图、温度场、资源分布、种群曲线
- 温度梯度 + 季节循环 + 温室效应 + 火山热点
- 涌现式食物网：消费者捕食生产者，资源竞争
- 自然选择：不适应环境的种群灭绝，适应者扩张

### 后续 Phase 概要
- **Phase 2**: AI 演化引擎 (OpenAI/Claude Provider，每 60 tick 调用 AI 决定突变方向)
- **Phase 3**: Web 前端 (FastAPI + WebSocket + React + Canvas)
- **Phase 4**: 实验框架 (SQLite 持久化、时间线回放、A/B 对比)
- **Phase 5**: 灾难系统、UI 美化、性能优化
