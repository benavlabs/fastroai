"""Cost calculator for AI model usage.

This module provides precise cost calculation using integer microcents
to avoid floating-point precision errors that can accumulate in billing.

Why microcents?
    Floating-point math has precision errors:
    >>> 0.1 + 0.2
    0.30000000000000004

    With integers, precision is exact:
    >>> 100 + 200
    300

    For billing systems, this matters.

Pricing format:
    Costs are stored in microcents per 1000 tokens.

    1 microcent = 1/10,000 cent = 1/1,000,000 dollar

    To convert from provider pricing ($/1M tokens) to microcents per 1K::

        microcents_per_1K = price_per_1M_dollars * 100

    Example: $2.50/1M tokens = 250 microcents per 1K tokens
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("fastroai.usage")

DEFAULT_PRICING: dict[str, dict[str, int]] = {
    # OpenAI
    "gpt-4o": {"input_cost_per_1k_tokens": 250, "output_cost_per_1k_tokens": 1000},
    "gpt-4o-mini": {"input_cost_per_1k_tokens": 15, "output_cost_per_1k_tokens": 60},
    "gpt-4-turbo": {"input_cost_per_1k_tokens": 1000, "output_cost_per_1k_tokens": 3000},
    "gpt-4": {"input_cost_per_1k_tokens": 3000, "output_cost_per_1k_tokens": 6000},
    "gpt-3.5-turbo": {"input_cost_per_1k_tokens": 50, "output_cost_per_1k_tokens": 150},
    "o1": {"input_cost_per_1k_tokens": 1500, "output_cost_per_1k_tokens": 6000},
    "o1-mini": {"input_cost_per_1k_tokens": 300, "output_cost_per_1k_tokens": 1200},
    "o1-preview": {"input_cost_per_1k_tokens": 1500, "output_cost_per_1k_tokens": 6000},
    # Anthropic
    "claude-sonnet-4-20250514": {"input_cost_per_1k_tokens": 300, "output_cost_per_1k_tokens": 1500},
    "claude-3-5-sonnet": {"input_cost_per_1k_tokens": 300, "output_cost_per_1k_tokens": 1500},
    "claude-3-5-sonnet-20241022": {"input_cost_per_1k_tokens": 300, "output_cost_per_1k_tokens": 1500},
    "claude-3-5-sonnet-latest": {"input_cost_per_1k_tokens": 300, "output_cost_per_1k_tokens": 1500},
    "claude-3-opus": {"input_cost_per_1k_tokens": 1500, "output_cost_per_1k_tokens": 7500},
    "claude-3-opus-20240229": {"input_cost_per_1k_tokens": 1500, "output_cost_per_1k_tokens": 7500},
    "claude-3-haiku": {"input_cost_per_1k_tokens": 25, "output_cost_per_1k_tokens": 125},
    "claude-3-haiku-20240307": {"input_cost_per_1k_tokens": 25, "output_cost_per_1k_tokens": 125},
    # Google
    "gemini-1.5-pro": {"input_cost_per_1k_tokens": 125, "output_cost_per_1k_tokens": 500},
    "gemini-1.5-flash": {"input_cost_per_1k_tokens": 7, "output_cost_per_1k_tokens": 30},
    "gemini-2.0-flash": {"input_cost_per_1k_tokens": 10, "output_cost_per_1k_tokens": 40},
    "gemini-2.0-flash-exp": {"input_cost_per_1k_tokens": 0, "output_cost_per_1k_tokens": 0},
    # Groq
    "llama-3.3-70b-versatile": {"input_cost_per_1k_tokens": 59, "output_cost_per_1k_tokens": 79},
    "llama-3.1-8b-instant": {"input_cost_per_1k_tokens": 5, "output_cost_per_1k_tokens": 8},
    "mixtral-8x7b-32768": {"input_cost_per_1k_tokens": 24, "output_cost_per_1k_tokens": 24},
}


class CostCalculator:
    """Token cost calculator with microcents precision.

    Uses integer arithmetic to avoid floating-point precision errors
    that can compound in billing systems.

    1 microcent = 1/10,000 cent = 1/1,000,000 dollar

    Example:
        calc = CostCalculator()

        # Calculate cost for a request
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # Input:  1000 * 250 / 1000 = 250 microcents
        # Output: 500 * 1000 / 1000 = 500 microcents
        # Total: 750 microcents = $0.00075

        print(f"Cost: {cost} microcents")
        print(f"Cost: ${calc.microcents_to_dollars(cost):.6f}")

        # Custom pricing
        my_pricing = DEFAULT_PRICING.copy()
        my_pricing["my-model"] = {
            "input_cost_per_1k_tokens": 500,
            "output_cost_per_1k_tokens": 1500,
        }
        calc = CostCalculator(pricing=my_pricing)
    """

    def __init__(self, pricing: dict[str, dict[str, int]] | None = None) -> None:
        """Initialize calculator with pricing data.

        Args:
            pricing: Custom pricing dict mapping model names to costs.
                    Defaults to DEFAULT_PRICING if not provided.
        """
        self.pricing = pricing if pricing is not None else DEFAULT_PRICING.copy()

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> int:
        """Calculate cost in microcents.

        Uses integer arithmetic with ceiling division to ensure precision
        and avoid small token counts rounding to zero.

        Args:
            model: Model identifier (e.g., "gpt-4o" or "openai:gpt-4o").
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.

        Returns:
            Cost in microcents (integer). Returns 0 for unknown models.

        Example:
            >>> calc.calculate_cost("gpt-4o", 1000, 500)
            750
        """
        normalized = self._normalize_model_name(model)
        model_pricing = self.pricing.get(normalized)

        if not model_pricing:
            logger.debug(f"No pricing for model '{model}' (normalized: '{normalized}')")
            return 0

        input_cost_per_1k = model_pricing.get("input_cost_per_1k_tokens", 0)
        output_cost_per_1k = model_pricing.get("output_cost_per_1k_tokens", 0)

        total_cost_unscaled = input_tokens * input_cost_per_1k + output_tokens * output_cost_per_1k
        if total_cost_unscaled == 0:
            return 0
        return (total_cost_unscaled + 999) // 1000

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for pricing lookup.

        Handles provider prefixes and case normalization.

        Args:
            model: Raw model identifier.

        Returns:
            Normalized model name.

        Examples:
            "openai:gpt-4o" -> "gpt-4o"
            "GPT-4o" -> "gpt-4o"
            "anthropic:claude-3-opus" -> "claude-3-opus"
        """
        if not model:
            return ""

        if ":" in model:
            model = model.split(":", 1)[1]

        return model.lower()

    def microcents_to_dollars(self, microcents: int) -> float:
        """Convert microcents to dollars for display.

        Use this only for display purposes. For calculations,
        always use integer microcents.

        Args:
            microcents: Cost in microcents.

        Returns:
            Cost in dollars (float).
        """
        return microcents / 1_000_000

    def dollars_to_microcents(self, dollars: float) -> int:
        """Convert dollars to microcents.

        Args:
            dollars: Cost in dollars.

        Returns:
            Cost in microcents (integer).
        """
        return round(dollars * 1_000_000)

    def format_cost(self, microcents: int) -> dict[str, Any]:
        """Format cost in multiple representations.

        Args:
            microcents: Cost in microcents.

        Returns:
            Dict with microcents, cents, and dollars representations.
        """
        return {
            "microcents": microcents,
            "cents": microcents // 10000,
            "dollars": self.microcents_to_dollars(microcents),
        }

    def get_model_pricing(self, model: str) -> dict[str, int] | None:
        """Get pricing for a specific model.

        Args:
            model: Model identifier.

        Returns:
            Pricing dict or None if model not found.
        """
        normalized = self._normalize_model_name(model)
        return self.pricing.get(normalized)

    def add_model_pricing(
        self,
        model: str,
        input_cost_per_1k_tokens: int,
        output_cost_per_1k_tokens: int,
    ) -> None:
        """Add or update pricing for a model.

        Args:
            model: Model identifier (will be normalized).
            input_cost_per_1k_tokens: Input cost in microcents per 1K tokens.
            output_cost_per_1k_tokens: Output cost in microcents per 1K tokens.
        """
        normalized = self._normalize_model_name(model)
        self.pricing[normalized] = {
            "input_cost_per_1k_tokens": input_cost_per_1k_tokens,
            "output_cost_per_1k_tokens": output_cost_per_1k_tokens,
        }
