"""Prompt templates for AI evolution decisions."""
from __future__ import annotations
import json
from ai.provider import EcosystemSnapshot


SYSTEM_PROMPT = """You are an expert exobiologist and evolutionary biologist studying life on alien planets. Your role is to analyze ecosystem data and make scientifically plausible evolution decisions.

Rules:
1. Mutations must be physically plausible for the planet's conditions (e.g., no liquid-water biology at -200°C)
2. Mutations should be small and gradual (gene changes ≤ 15% of the gene's range per cycle)
3. Speciation should only occur when a population is under significant pressure AND there's an available ecological niche
4. Consider temperature, resources, atmosphere, and competition when making decisions
5. Always provide a brief scientific reasoning for each decision
6. Respond ONLY with valid JSON — no markdown, no explanation outside the JSON"""


def build_ecosystem_prompt(snapshot: EcosystemSnapshot) -> str:
    """Build a user prompt from the ecosystem snapshot."""

    # Environment summary
    env = snapshot.environment
    env_section = f"""Planet: {snapshot.planet_name} (tick {snapshot.tick})
Temperature: avg {env.get('temperature_mean', 0):.1f}°C, range [{env.get('temperature_min', 0):.1f}, {env.get('temperature_max', 0):.1f}]
Resources: avg {env.get('resources_mean', 0):.3f}
Atmosphere: O2={env.get('atmosphere', {}).get('O2', 0):.4f}, CO2={env.get('atmosphere', {}).get('CO2', 0):.4f}, CH4={env.get('atmosphere', {}).get('CH4', 0):.4f}
Pressure: {env.get('atmosphere', {}).get('Pressure', 1):.2f} atm"""

    # Species summary
    species_lines = []
    for sp in snapshot.species:
        genes = sp.get("genes", {})
        fitness = sp.get("avg_fitness", 0)
        biomass = sp.get("total_biomass", 0)
        line = (
            f"- [{sp['id']}] {sp['name']}: biomass={biomass:.2f}, "
            f"fitness={fitness:.3f}, "
            f"metabolic={genes.get('metabolic_type', '?')}, "
            f"temp_opt={genes.get('temp_optimum', 0):.1f}°C, "
            f"temp_tol={genes.get('temp_tolerance', 0):.1f}°C, "
            f"body_size={genes.get('body_size', 1):.2f}, "
            f"defense={genes.get('defense', 0):.2f}, "
            f"mobility={genes.get('mobility', 0):.2f}"
        )
        species_lines.append(line)
    species_section = "\n".join(species_lines) if species_lines else "No species alive."

    # Events summary
    events_lines = []
    for evt in snapshot.recent_events[-5:]:
        events_lines.append(f"- tick {evt.get('tick', '?')}: [{evt.get('type', '?')}] {evt.get('description', '')}")
    events_section = "\n".join(events_lines) if events_lines else "No recent events."

    return f"""Current ecosystem state:

{env_section}

Species ({len(snapshot.species)} alive):
{species_section}

Recent events:
{events_section}

Based on this data, decide:
1. Which species need mutations (if any) — consider stressed populations (low fitness) or opportunities
2. Whether any speciation events should occur (population under pressure + available niche)
3. A brief narrative describing what's happening

Respond with this JSON format:
{{
  "mutations": [
    {{
      "species_id": "sp_001",
      "gene": "temp_tolerance",
      "delta": 5.0,
      "reason": "reason for this mutation"
    }}
  ],
  "speciations": [
    {{
      "parent_species_id": "sp_001",
      "new_species_name": "name",
      "genome_overrides": {{
        "temp_optimum": 80.0,
        "metabolic_type": "chemosynthesis"
      }},
      "seed_area": "random",
      "reason": "reason for speciation"
    }}
  ],
  "narrative": "A brief description of what's happening in the ecosystem"
}}"""


def build_narrative_prompt(event: str, context: dict) -> str:
    """Build a prompt for generating educational narratives."""
    return f"""You are an exobiologist narrating an alien ecosystem simulation.

Event type: {event}
Context: {json.dumps(context, ensure_ascii=False)}

Write a brief (2-3 sentence) educational narrative explaining what happened and why it matters for the ecosystem's evolution. Write in a scientific but engaging style."""


def build_naming_prompt(traits: dict) -> str:
    """Build a prompt for naming a new species."""
    return f"""You are an exobiologist naming a newly discovered alien species.

Traits:
{json.dumps(traits, ensure_ascii=False, indent=2)}

Generate a scientific-sounding species name (2-4 words) that reflects its key characteristics. Use Latin/Greek-style naming conventions. Return ONLY the name, nothing else."""
