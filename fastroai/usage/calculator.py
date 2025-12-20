"""Cost calculator for AI model usage.

This module provides precise cost calculation using integer microcents
to avoid floating-point precision errors that can accumulate in billing.

Uses genai-prices package for up-to-date model pricing data, with support
for custom pricing overrides (e.g., volume discounts).

Why microcents?
    Floating-point math has precision errors:
    >>> 0.1 + 0.2
    0.30000000000000004

    With integers, precision is exact:
    >>> 100 + 200
    300

    For billing systems, this matters.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from genai_prices import Usage, calc_price

logger = logging.getLogger("fastroai.usage")


class CostCalculator:
    """Token cost calculator with microcents precision.

    Uses genai-prices for model pricing data, with support for custom
    pricing overrides. All costs are returned as integer microcents.

    Supports cache tokens (90% cheaper on Anthropic) and audio tokens
    for accurate cost calculation with prompt caching and multimodal models.

    1 microcent = 1/10,000 cent = 1/1,000,000 dollar

    Examples:
        ```python
        calc = CostCalculator()

        # Calculate cost for a request
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        print(f"Cost: {cost} microcents")
        print(f"Cost: ${calc.microcents_to_dollars(cost):.6f}")

        # With cache tokens (saves ~18% on this example)
        cost = calc.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=250,
            output_tokens=150,
            cache_read_tokens=200,  # 200 of 250 input tokens were cached
        )

        # With custom pricing override (e.g., volume discount)
        calc = CostCalculator(pricing_overrides={
            "gpt-4o": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
        })
        ```
    """

    def __init__(
        self,
        pricing_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize calculator.

        Args:
            pricing_overrides: Custom pricing for specific models. Keys are model
                names, values are dicts with 'input_per_mtok', 'output_per_mtok',
                and optionally 'cache_read_per_mtok', 'cache_write_per_mtok'
                (dollars per million tokens). Use for volume discounts or custom models.
        """
        self._overrides = pricing_overrides or {}

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        input_audio_tokens: int = 0,
        output_audio_tokens: int = 0,
        cache_audio_read_tokens: int = 0,
    ) -> int:
        """Calculate cost in microcents.

        Supports cache tokens for accurate cost calculation when prompt caching
        is enabled. Cache read tokens are typically 90% cheaper on Anthropic.

        Args:
            model: Model identifier (e.g., "gpt-4o" or "openai:gpt-4o").
            input_tokens: Number of input/prompt tokens (includes cached + uncached).
            output_tokens: Number of output/completion tokens.
            cache_read_tokens: Tokens read from prompt cache (typically 90% cheaper).
            cache_write_tokens: Tokens written to prompt cache (typically 25% premium).
            input_audio_tokens: Audio input tokens for multimodal models.
            output_audio_tokens: Audio output tokens for multimodal models.
            cache_audio_read_tokens: Audio tokens read from cache.

        Returns:
            Cost in microcents (integer). Returns 0 for unknown models.

        Examples:
            ```python
            calc = CostCalculator()

            # Basic usage
            cost = calc.calculate_cost("gpt-4o", 1000, 500)
            print(f"${calc.microcents_to_dollars(cost):.6f}")

            # With cache tokens (more accurate cost)
            cost = calc.calculate_cost(
                "claude-3-5-sonnet",
                input_tokens=250,
                output_tokens=150,
                cache_read_tokens=200,
            )
            ```
        """
        normalized = self._normalize_model_name(model)

        if normalized in self._overrides:
            return self._calc_from_override(
                normalized,
                input_tokens,
                output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )

        try:
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens if cache_read_tokens else None,
                cache_write_tokens=cache_write_tokens if cache_write_tokens else None,
                input_audio_tokens=input_audio_tokens if input_audio_tokens else None,
                output_audio_tokens=output_audio_tokens if output_audio_tokens else None,
                cache_audio_read_tokens=(cache_audio_read_tokens if cache_audio_read_tokens else None),
            )
            price = calc_price(usage, model_ref=normalized)
            return self._dollars_to_microcents(price.total_price)
        except LookupError:
            logger.debug(f"No pricing found for model '{model}' (normalized: '{normalized}')")
            return 0

    def _calc_from_override(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> int:
        """Calculate cost using override pricing with cache token support.

        Cache tokens are priced using explicit rates if provided, otherwise:
        - cache_read: 10% of input rate (90% discount, matching Anthropic)
        - cache_write: 125% of input rate (25% premium)
        """
        override = self._overrides[model]
        input_per_mtok = Decimal(str(override.get("input_per_mtok", 0)))
        output_per_mtok = Decimal(str(override.get("output_per_mtok", 0)))

        cache_read_per_mtok = Decimal(str(override.get("cache_read_per_mtok", 0)))
        if not cache_read_per_mtok and cache_read_tokens:
            cache_read_per_mtok = input_per_mtok * Decimal("0.1")

        cache_write_per_mtok = Decimal(str(override.get("cache_write_per_mtok", 0)))
        if not cache_write_per_mtok and cache_write_tokens:
            cache_write_per_mtok = input_per_mtok * Decimal("1.25")

        uncached_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)

        mtok = Decimal(1_000_000)
        input_cost = Decimal(uncached_input) / mtok * input_per_mtok
        output_cost = Decimal(output_tokens) / mtok * output_per_mtok
        cache_read_cost = Decimal(cache_read_tokens) / mtok * cache_read_per_mtok
        cache_write_cost = Decimal(cache_write_tokens) / mtok * cache_write_per_mtok

        total_dollars = input_cost + output_cost + cache_read_cost + cache_write_cost
        return self._dollars_to_microcents(total_dollars)

    def _dollars_to_microcents(self, dollars: Decimal) -> int:
        """Convert Decimal dollars to int microcents."""
        return int(dollars * 1_000_000)

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for pricing lookup.

        Handles provider prefixes (e.g., "openai:gpt-4o" -> "gpt-4o").

        Args:
            model: Raw model identifier.

        Returns:
            Normalized model name.

        Examples:
            ```python
            calc = CostCalculator()
            calc._normalize_model_name("openai:gpt-4o")  # -> "gpt-4o"
            calc._normalize_model_name("anthropic:claude-3-opus")  # -> "claude-3-opus"
            ```
        """
        if not model:
            return ""

        if ":" in model:
            model = model.split(":", 1)[1]

        return model

    def microcents_to_dollars(self, microcents: int) -> float:
        """Convert microcents to dollars for display.

        Use this only for display purposes. For calculations,
        always use integer microcents.

        Args:
            microcents: Cost in microcents.

        Returns:
            Cost in dollars (float).

        Examples:
            ```python
            calc = CostCalculator()
            dollars = calc.microcents_to_dollars(1_500_000)
            print(f"${dollars:.2f}")  # $1.50
            ```
        """
        return microcents / 1_000_000

    def dollars_to_microcents(self, dollars: float) -> int:
        """Convert dollars to microcents.

        Args:
            dollars: Cost in dollars.

        Returns:
            Cost in microcents (integer).

        Examples:
            ```python
            calc = CostCalculator()

            # Set a budget of $0.10
            budget = calc.dollars_to_microcents(0.10)
            print(budget)  # 100000
            ```
        """
        return round(dollars * 1_000_000)

    def format_cost(self, microcents: int) -> dict[str, Any]:
        """Format cost in multiple representations.

        Args:
            microcents: Cost in microcents.

        Returns:
            Dict with microcents, cents, and dollars representations.

        Examples:
            ```python
            calc = CostCalculator()
            formatted = calc.format_cost(1_500_000)

            print(formatted)
            # {
            #     "microcents": 1500000,
            #     "cents": 150,
            #     "dollars": 1.5
            # }
            ```
        """
        return {
            "microcents": microcents,
            "cents": microcents // 10000,
            "dollars": self.microcents_to_dollars(microcents),
        }

    def add_pricing_override(
        self,
        model: str,
        input_per_mtok: float,
        output_per_mtok: float,
        *,
        cache_read_per_mtok: float | None = None,
        cache_write_per_mtok: float | None = None,
    ) -> None:
        """Add or update pricing override for a model.

        Use this for custom pricing (volume discounts) or models not in genai-prices.

        Args:
            model: Model identifier (will be normalized).
            input_per_mtok: Input cost in dollars per million tokens.
            output_per_mtok: Output cost in dollars per million tokens.
            cache_read_per_mtok: Cache read cost (default: 10% of input, i.e. 90% discount).
            cache_write_per_mtok: Cache write cost (default: 125% of input, i.e. 25% premium).

        Examples:
            ```python
            calc = CostCalculator()

            # Add volume discount pricing (20% off standard)
            calc.add_pricing_override(
                model="gpt-4o",
                input_per_mtok=2.00,   # Standard is $2.50
                output_per_mtok=8.00,  # Standard is $10.00
            )

            # Add custom model with explicit cache pricing
            calc.add_pricing_override(
                model="my-cached-model",
                input_per_mtok=3.00,
                output_per_mtok=15.00,
                cache_read_per_mtok=0.30,   # 90% discount
                cache_write_per_mtok=3.75,  # 25% premium
            )
            ```
        """
        normalized = self._normalize_model_name(model)
        override: dict[str, float] = {
            "input_per_mtok": input_per_mtok,
            "output_per_mtok": output_per_mtok,
        }
        if cache_read_per_mtok is not None:
            override["cache_read_per_mtok"] = cache_read_per_mtok
        if cache_write_per_mtok is not None:
            override["cache_write_per_mtok"] = cache_write_per_mtok
        self._overrides[normalized] = override
