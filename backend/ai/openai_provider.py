"""OpenAI provider for AI evolution decisions."""
from __future__ import annotations
import json
import re
import logging
from openai import AsyncOpenAI
from ai.provider import (
    AIProvider, EcosystemSnapshot, EvolutionDecision,
    MutationDecision, SpeciationDecision,
)
from ai.prompts import SYSTEM_PROMPT, build_ecosystem_prompt, build_narrative_prompt, build_naming_prompt

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """AI provider using OpenAI's API (GPT-4o-mini by default)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)

    async def analyze_ecosystem(self, snapshot: EcosystemSnapshot) -> EvolutionDecision:
        user_prompt = build_ecosystem_prompt(snapshot)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_decision(raw)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return EvolutionDecision(narrative=f"[AI unavailable: {e}]")

    async def generate_narrative(self, event: str, context: dict) -> str:
        prompt = build_narrative_prompt(event, context)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=300,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI narrative error: {e}")
            return f"[Narrative unavailable: {e}]"

    async def name_species(self, traits: dict) -> str:
        prompt = build_naming_prompt(traits)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=50,
            )
            name = (response.choices[0].message.content or "").strip().strip('"')
            return name if name else "Unknown Species"
        except Exception as e:
            logger.error(f"OpenAI naming error: {e}")
            return "Unknown Species"

    @staticmethod
    def _parse_decision(raw: str) -> EvolutionDecision:
        """Parse LLM response into an EvolutionDecision."""
        # Try to extract JSON from the response
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            return EvolutionDecision(narrative=raw, raw_response=raw)

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return EvolutionDecision(narrative=raw, raw_response=raw)

        mutations = []
        for m in data.get("mutations", []):
            try:
                mutations.append(MutationDecision(
                    species_id=m["species_id"],
                    gene=m["gene"],
                    delta=float(m["delta"]),
                    reason=m.get("reason", ""),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid mutation: {e}")

        speciations = []
        for s in data.get("speciations", []):
            try:
                speciations.append(SpeciationDecision(
                    parent_species_id=s["parent_species_id"],
                    new_species_name=s.get("new_species_name", "Unknown"),
                    genome_overrides=s.get("genome_overrides", {}),
                    seed_area=s.get("seed_area", "random"),
                    reason=s.get("reason", ""),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid speciation: {e}")

        return EvolutionDecision(
            mutations=mutations,
            speciations=speciations,
            narrative=data.get("narrative", ""),
            raw_response=raw,
        )
