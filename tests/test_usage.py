"""Tests for the usage module."""

import pytest

from fastroai.usage import CostCalculator


class TestCostCalculator:
    """Tests for CostCalculator with genai-prices."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        """Create a CostCalculator."""
        return CostCalculator()

    def test_calculate_cost_gpt4o(self, calc: CostCalculator) -> None:
        """Test cost calculation for GPT-4o."""
        # gpt-4o: $2.50/1M input, $10/1M output (from genai-prices)
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # 1000 / 1M * $2.50 = $0.0025 = 2500 microcents
        # 500 / 1M * $10 = $0.005 = 5000 microcents
        # Total: 7500 microcents
        assert cost == 7500

    def test_calculate_cost_gpt4o_mini(self, calc: CostCalculator) -> None:
        """Test cost calculation for GPT-4o-mini."""
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = calc.calculate_cost("gpt-4o-mini", input_tokens=10000, output_tokens=1000)
        # 10000 / 1M * $0.15 = $0.0015 = 1500 microcents
        # 1000 / 1M * $0.60 = $0.0006 = 600 microcents
        # Total: 2100 microcents
        assert cost == 2100

    def test_calculate_cost_claude(self, calc: CostCalculator) -> None:
        """Test cost calculation for Claude models."""
        # claude-3-5-sonnet: $3/1M input, $15/1M output
        cost = calc.calculate_cost("claude-3-5-sonnet", input_tokens=2000, output_tokens=1000)
        # 2000 / 1M * $3 = $0.006 = 6000 microcents
        # 1000 / 1M * $15 = $0.015 = 15000 microcents
        # Total: 21000 microcents
        assert cost == 21000

    def test_calculate_cost_with_provider_prefix(self, calc: CostCalculator) -> None:
        """Should handle provider prefix in model name."""
        cost_with_prefix = calc.calculate_cost("openai:gpt-4o", input_tokens=1000, output_tokens=500)
        cost_without = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost_with_prefix == cost_without

    def test_calculate_cost_unknown_model(self, calc: CostCalculator) -> None:
        """Should return 0 for unknown models."""
        cost = calc.calculate_cost("unknown-model-xyz-123", input_tokens=1000, output_tokens=500)
        assert cost == 0

    def test_calculate_cost_empty_model(self, calc: CostCalculator) -> None:
        """Should return 0 for empty model name."""
        cost = calc.calculate_cost("", input_tokens=1000, output_tokens=500)
        assert cost == 0

    def test_calculate_cost_zero_tokens(self, calc: CostCalculator) -> None:
        """Should return 0 for zero tokens."""
        cost = calc.calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0


class TestCostCalculatorOverrides:
    """Tests for pricing overrides."""

    def test_pricing_override_takes_precedence(self) -> None:
        """Override pricing should be used instead of genai-prices."""
        # Override gpt-4o with custom pricing (e.g., volume discount)
        calc = CostCalculator(pricing_overrides={"gpt-4o": {"input_per_mtok": 2.00, "output_per_mtok": 8.00}})
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # 1000 / 1M * $2.00 = $0.002 = 2000 microcents
        # 500 / 1M * $8.00 = $0.004 = 4000 microcents
        # Total: 6000 microcents (less than standard 7500)
        assert cost == 6000

    def test_add_pricing_override(self) -> None:
        """Should add new model pricing override."""
        calc = CostCalculator()
        calc.add_pricing_override(
            model="my-custom-model",
            input_per_mtok=1.00,
            output_per_mtok=2.00,
        )

        cost = calc.calculate_cost("my-custom-model", input_tokens=1000, output_tokens=1000)
        # 1000 / 1M * $1.00 = $0.001 = 1000 microcents
        # 1000 / 1M * $2.00 = $0.002 = 2000 microcents
        # Total: 3000 microcents
        assert cost == 3000

    def test_add_pricing_override_overrides_genai_prices(self) -> None:
        """Override should take precedence over genai-prices."""
        calc = CostCalculator()
        original_cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        assert original_cost > 0

        calc.add_pricing_override("gpt-4o", input_per_mtok=0.50, output_per_mtok=0)
        new_cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)

        assert new_cost != original_cost
        # 1000 / 1M * $0.50 = 500 microcents
        assert new_cost == 500

    def test_override_normalizes_model_name(self) -> None:
        """Override should work with provider prefix."""
        calc = CostCalculator(pricing_overrides={"gpt-4o": {"input_per_mtok": 1.00, "output_per_mtok": 2.00}})

        cost_with_prefix = calc.calculate_cost("openai:gpt-4o", input_tokens=1000000, output_tokens=0)
        cost_without = calc.calculate_cost("gpt-4o", input_tokens=1000000, output_tokens=0)

        assert cost_with_prefix == cost_without
        assert cost_with_prefix == 1_000_000  # $1.00


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


class TestCostCalculatorProviders:
    """Tests for various model providers via genai-prices."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        return CostCalculator()

    def test_openai_models(self, calc: CostCalculator) -> None:
        """Should have pricing for OpenAI models."""
        assert calc.calculate_cost("gpt-4o", 1000, 0) > 0
        assert calc.calculate_cost("gpt-4o-mini", 1000, 0) > 0
        assert calc.calculate_cost("gpt-4-turbo", 1000, 0) > 0

    def test_anthropic_models(self, calc: CostCalculator) -> None:
        """Should have pricing for Anthropic models."""
        assert calc.calculate_cost("claude-3-5-sonnet", 1000, 0) > 0
        assert calc.calculate_cost("claude-3-opus", 1000, 0) > 0
        assert calc.calculate_cost("claude-3-haiku", 1000, 0) > 0

    def test_google_models(self, calc: CostCalculator) -> None:
        """Should have pricing for Google models."""
        assert calc.calculate_cost("gemini-1.5-pro", 1000, 0) > 0
        assert calc.calculate_cost("gemini-1.5-flash", 1000, 0) > 0

    def test_relative_pricing_makes_sense(self, calc: CostCalculator) -> None:
        """Sanity check: larger models should cost more than smaller ones."""
        gpt4o = calc.calculate_cost("gpt-4o", 1000, 1000)
        gpt4o_mini = calc.calculate_cost("gpt-4o-mini", 1000, 1000)
        assert gpt4o > gpt4o_mini

        opus = calc.calculate_cost("claude-3-opus", 1000, 1000)
        haiku = calc.calculate_cost("claude-3-haiku", 1000, 1000)
        assert opus > haiku


class TestCostCalculatorPrecision:
    """Tests for precision and integer arithmetic."""

    def test_returns_integer(self) -> None:
        """Should always return integer microcents."""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, int)

    def test_large_token_counts(self) -> None:
        """Should handle large token counts correctly."""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        # 1M / 1M * $2.50 = $2.50 = 2_500_000 microcents
        # 500K / 1M * $10 = $5.00 = 5_000_000 microcents
        # Total: 7_500_000 microcents = $7.50
        assert cost == 7_500_000
        assert calc.microcents_to_dollars(cost) == 7.5

    def test_override_precision(self) -> None:
        """Override calculations should maintain precision."""
        calc = CostCalculator(pricing_overrides={"test-model": {"input_per_mtok": 0.001, "output_per_mtok": 0.002}})

        cost = calc.calculate_cost("test-model", input_tokens=1_000_000, output_tokens=1_000_000)
        # 1M / 1M * $0.001 = $0.001 = 1000 microcents
        # 1M / 1M * $0.002 = $0.002 = 2000 microcents
        # Total: 3000 microcents
        assert cost == 3000
