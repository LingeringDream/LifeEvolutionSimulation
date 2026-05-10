"""Factory for creating AI providers."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from ai.provider import AIProvider

# Load .env from backend/ directory
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


def create_ai_provider(
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> AIProvider:
    """Create an AI provider instance.

    Supported providers:
        - "openai": OpenAI API (GPT-4o-mini, etc.)
        - "claude": Anthropic Claude API
        - "custom": Any OpenAI-compatible API (DeepSeek, Moonshot, Ollama, vLLM, etc.)

    API key resolution order:
    1. Explicit `api_key` argument
    2. Environment variable (OPENAI_API_KEY / ANTHROPIC_API_KEY / CUSTOM_API_KEY)
    3. .env file in backend/ directory

    Args:
        provider: "openai", "claude", or "custom".
        api_key: API key override.
        model: Model name override.
        base_url: Custom API base URL (required for "custom", optional for others).

    Returns:
        An AIProvider instance.

    Raises:
        ValueError: If provider name is unknown or custom provider lacks base_url.
    """
    provider = provider.lower()

    if provider == "openai":
        from ai.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
            base_url=base_url,
        )
    elif provider == "claude":
        from ai.claude_provider import ClaudeProvider
        return ClaudeProvider(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=model or "claude-sonnet-4-20250514",
        )
    elif provider == "custom":
        resolved_url = base_url or os.environ.get("CUSTOM_API_BASE_URL")
        if not resolved_url:
            raise ValueError(
                "Custom provider requires --ai-base-url or CUSTOM_API_BASE_URL in .env"
            )
        from ai.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key or os.environ.get("CUSTOM_API_KEY", "no-key"),
            model=model or "default",
            base_url=resolved_url,
        )
    else:
        raise ValueError(f"Unknown AI provider: {provider}. Use 'openai', 'claude', or 'custom'.")
