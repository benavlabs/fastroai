"""Tests for the usage module."""

import pytest

from fastroai.usage import DEFAULT_PRICING, CostCalculator


class TestDefaultPricing:
    """Tests for DEFAULT_PRICING data."""

    def test_has_openai_models(self) -> None:
        """Should include common OpenAI models."""
        assert "gpt-4o" in DEFAULT_PRICING
        assert "gpt-4o-mini" in DEFAULT_PRICING
        assert "gpt-4-turbo" in DEFAULT_PRICING

    def test_has_anthropic_models(self) -> None:
        """Should include common Anthropic models."""
        assert "claude-3-5-sonnet" in DEFAULT_PRICING
        assert "claude-3-opus" in DEFAULT_PRICING
        assert "claude-3-haiku" in DEFAULT_PRICING

    def test_has_google_models(self) -> None:
        """Should include common Google models."""
        assert "gemini-1.5-pro" in DEFAULT_PRICING
        assert "gemini-1.5-flash" in DEFAULT_PRICING

    def test_pricing_structure(self) -> None:
        """Each model should have input and output costs."""
        for model, pricing in DEFAULT_PRICING.items():
            assert "input_cost_per_1k_tokens" in pricing, f"{model} missing input cost"
            assert "output_cost_per_1k_tokens" in pricing, f"{model} missing output cost"
            assert isinstance(pricing["input_cost_per_1k_tokens"], int)
            assert isinstance(pricing["output_cost_per_1k_tokens"], int)


class TestCostCalculator:
    """Tests for CostCalculator."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        """Create a CostCalculator with default pricing."""
        return CostCalculator()

    def test_calculate_cost_gpt4o(self, calc: CostCalculator) -> None:
        """Test cost calculation for GPT-4o."""
        # gpt-4o: 250 microcents/1K input, 1000 microcents/1K output
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # 1000 * 250 / 1000 = 250
        # 500 * 1000 / 1000 = 500
        # Total: 750
        assert cost == 750

    def test_calculate_cost_gpt4o_mini(self, calc: CostCalculator) -> None:
        """Test cost calculation for GPT-4o-mini."""
        # gpt-4o-mini: 15 microcents/1K input, 60 microcents/1K output
        cost = calc.calculate_cost("gpt-4o-mini", input_tokens=10000, output_tokens=1000)
        # 10000 * 15 / 1000 = 150
        # 1000 * 60 / 1000 = 60
        # Total: 210
        assert cost == 210

    def test_calculate_cost_claude(self, calc: CostCalculator) -> None:
        """Test cost calculation for Claude models."""
        # claude-3-5-sonnet: 300 microcents/1K input, 1500 microcents/1K output
        cost = calc.calculate_cost("claude-3-5-sonnet", input_tokens=2000, output_tokens=1000)
        # 2000 * 300 / 1000 = 600
        # 1000 * 1500 / 1000 = 1500
        # Total: 2100
        assert cost == 2100

    def test_calculate_cost_with_provider_prefix(self, calc: CostCalculator) -> None:
        """Should handle provider prefix in model name."""
        cost_with_prefix = calc.calculate_cost("openai:gpt-4o", input_tokens=1000, output_tokens=500)
        cost_without = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost_with_prefix == cost_without

    def test_calculate_cost_case_insensitive(self, calc: CostCalculator) -> None:
        """Should handle case variations in model name."""
        cost_upper = calc.calculate_cost("GPT-4O", input_tokens=1000, output_tokens=500)
        cost_lower = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost_upper == cost_lower

    def test_calculate_cost_unknown_model(self, calc: CostCalculator) -> None:
        """Should return 0 for unknown models."""
        cost = calc.calculate_cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)
        assert cost == 0

    def test_calculate_cost_empty_model(self, calc: CostCalculator) -> None:
        """Should return 0 for empty model name."""
        cost = calc.calculate_cost("", input_tokens=1000, output_tokens=500)
        assert cost == 0

    def test_calculate_cost_zero_tokens(self, calc: CostCalculator) -> None:
        """Should return 0 for zero tokens."""
        cost = calc.calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0

    def test_custom_pricing(self) -> None:
        """Should use custom pricing when provided."""
        custom_pricing = {
            "my-model": {
                "input_cost_per_1k_tokens": 100,
                "output_cost_per_1k_tokens": 200,
            }
        }
        calc = CostCalculator(pricing=custom_pricing)
        cost = calc.calculate_cost("my-model", input_tokens=1000, output_tokens=1000)
        # 1000 * 100 / 1000 + 1000 * 200 / 1000 = 100 + 200 = 300
        assert cost == 300

    def test_custom_pricing_doesnt_have_defaults(self) -> None:
        """Custom pricing should replace defaults, not merge."""
        custom_pricing = {"my-model": {"input_cost_per_1k_tokens": 100, "output_cost_per_1k_tokens": 200}}
        calc = CostCalculator(pricing=custom_pricing)
        # Should not find gpt-4o
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost == 0


class TestCostCalculatorConversions:
    """Tests for conversion methods."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        return CostCalculator()

    def test_microcents_to_dollars(self, calc: CostCalculator) -> None:
        """Test microcents to dollars conversion."""
        assert calc.microcents_to_dollars(1_000_000) == 1.0
        assert calc.microcents_to_dollars(100_000) == 0.1
        assert calc.microcents_to_dollars(750) == 0.00075

    def test_dollars_to_microcents(self, calc: CostCalculator) -> None:
        """Test dollars to microcents conversion."""
        assert calc.dollars_to_microcents(1.0) == 1_000_000
        assert calc.dollars_to_microcents(0.1) == 100_000
        assert calc.dollars_to_microcents(0.00075) == 750

    def test_format_cost(self, calc: CostCalculator) -> None:
        """Test cost formatting."""
        result = calc.format_cost(1_234_567)
        assert result["microcents"] == 1_234_567
        assert result["cents"] == 123  # 1_234_567 // 10000
        assert result["dollars"] == pytest.approx(1.234567)


class TestCostCalculatorModelManagement:
    """Tests for model pricing management."""

    def test_get_model_pricing_exists(self) -> None:
        """Should return pricing for known model."""
        calc = CostCalculator()
        pricing = calc.get_model_pricing("gpt-4o")
        assert pricing is not None
        assert "input_cost_per_1k_tokens" in pricing
        assert "output_cost_per_1k_tokens" in pricing

    def test_get_model_pricing_not_exists(self) -> None:
        """Should return None for unknown model."""
        calc = CostCalculator()
        pricing = calc.get_model_pricing("unknown-model")
        assert pricing is None

    def test_get_model_pricing_normalizes(self) -> None:
        """Should normalize model name when looking up."""
        calc = CostCalculator()
        pricing = calc.get_model_pricing("openai:GPT-4o")
        assert pricing is not None

    def test_add_model_pricing(self) -> None:
        """Should add new model pricing."""
        calc = CostCalculator()
        calc.add_model_pricing("my-new-model", input_cost_per_1k_tokens=50, output_cost_per_1k_tokens=100)

        cost = calc.calculate_cost("my-new-model", input_tokens=1000, output_tokens=1000)
        assert cost == 150

    def test_add_model_pricing_updates_existing(self) -> None:
        """Should update existing model pricing."""
        calc = CostCalculator()
        original_cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)

        calc.add_model_pricing("gpt-4o", input_cost_per_1k_tokens=999, output_cost_per_1k_tokens=0)
        new_cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)

        assert new_cost != original_cost
        assert new_cost == 999


class TestCostCalculatorPrecision:
    """Tests for precision and integer arithmetic."""

    def test_no_floating_point_errors(self) -> None:
        """Should avoid floating-point precision issues."""
        calc = CostCalculator()

        # Calculate many small costs
        total = 0
        for _ in range(10000):
            total += calc.calculate_cost("gpt-4o-mini", input_tokens=100, output_tokens=50)

        # 100 * 15 = 1500, 50 * 60 = 3000
        # (1500 + 3000 + 999) // 1000 = 5 microcents (ceiling of 4.5)
        # Total: 50000 microcents

        # With integer math, should be exactly 50000
        # With floating point, might have small errors
        assert total == 50000
        assert isinstance(total, int)

    def test_large_token_counts(self) -> None:
        """Should handle large token counts correctly."""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        # 1_000_000 * 250 / 1000 = 250_000
        # 500_000 * 1000 / 1000 = 500_000
        # Total: 750_000 microcents = $0.75
        assert cost == 750_000
        assert calc.microcents_to_dollars(cost) == 0.75
