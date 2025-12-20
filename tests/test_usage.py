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

    def test_override_with_cache_tokens_default_discount(self) -> None:
        """Override should apply default 90% discount for cache_read_tokens."""
        calc = CostCalculator(pricing_overrides={"my-model": {"input_per_mtok": 10.00, "output_per_mtok": 20.00}})

        # Without cache - all 1000 tokens at full price
        cost_no_cache = calc.calculate_cost("my-model", input_tokens=1000, output_tokens=0)
        # 1000 / 1M * $10.00 = $0.01 = 10000 microcents
        assert cost_no_cache == 10000

        # With cache - 800 cached at 10% rate, 200 at full rate
        cost_with_cache = calc.calculate_cost(
            "my-model",
            input_tokens=1000,
            output_tokens=0,
            cache_read_tokens=800,
        )
        # 200 uncached / 1M * $10.00 = $0.002 = 2000 microcents
        # 800 cached / 1M * $1.00 (10% of $10) = $0.0008 = 800 microcents
        # Total: 2800 microcents
        assert cost_with_cache == 2800
        assert cost_with_cache < cost_no_cache

    def test_override_with_explicit_cache_rates(self) -> None:
        """Override should use explicit cache rates when provided."""
        calc = CostCalculator()
        calc.add_pricing_override(
            model="my-model",
            input_per_mtok=10.00,
            output_per_mtok=20.00,
            cache_read_per_mtok=0.50,  # Custom rate, not default 10%
        )

        cost = calc.calculate_cost(
            "my-model",
            input_tokens=1000,
            output_tokens=0,
            cache_read_tokens=800,
        )
        # 200 uncached / 1M * $10.00 = 2000 microcents
        # 800 cached / 1M * $0.50 = 400 microcents
        # Total: 2400 microcents
        assert cost == 2400

    def test_override_with_cache_write_tokens(self) -> None:
        """Override should apply default 25% premium for cache_write_tokens."""
        calc = CostCalculator(pricing_overrides={"my-model": {"input_per_mtok": 10.00, "output_per_mtok": 20.00}})

        cost = calc.calculate_cost(
            "my-model",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        # 500 uncached / 1M * $10.00 = 5000 microcents
        # 500 cache_write / 1M * $12.50 (125% of $10) = 6250 microcents
        # Total: 11250 microcents
        assert cost == 11250

    def test_add_pricing_override_with_cache_write_rate(self) -> None:
        """add_pricing_override should accept cache_write_per_mtok."""
        calc = CostCalculator()
        calc.add_pricing_override(
            model="my-model",
            input_per_mtok=10.00,
            output_per_mtok=20.00,
            cache_write_per_mtok=15.00,  # Custom 50% premium instead of default 25%
        )

        cost = calc.calculate_cost(
            "my-model",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        # 500 uncached / 1M * $10.00 = 5000 microcents
        # 500 cache_write / 1M * $15.00 = 7500 microcents
        # Total: 12500 microcents
        assert cost == 12500


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


class TestCostCalculatorCacheTokens:
    """Tests for cache token pricing (prompt caching)."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        return CostCalculator()

    def test_cache_read_tokens_reduce_cost(self, calc: CostCalculator) -> None:
        """Cache read tokens should be cheaper than regular input tokens."""
        # Claude 3.5 Sonnet: $3/1M input, $0.30/1M cache read (90% cheaper)
        cost_no_cache = calc.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=1000,
            output_tokens=0,
        )

        cost_with_cache = calc.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=1000,
            output_tokens=0,
            cache_read_tokens=800,  # 800 of 1000 tokens cached
        )

        # Cache should reduce cost (800 tokens at 90% discount)
        assert cost_with_cache < cost_no_cache

    def test_cache_tokens_savings_example(self, calc: CostCalculator) -> None:
        """Verify the Sarah example from implementation journal."""
        # Personal Finance Assistant scenario:
        # - System prompt: 200 tokens (cached)
        # - User message: 50 tokens (uncached)
        # - Output: 150 tokens

        # Without cache awareness (old behavior - all 250 input at full price)
        cost_old_way = calc.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=250,
            output_tokens=150,
        )

        # With cache awareness (200 cached, 50 uncached)
        cost_new_way = calc.calculate_cost(
            "claude-3-5-sonnet",
            input_tokens=250,
            output_tokens=150,
            cache_read_tokens=200,
        )

        # New way should be cheaper due to cache discount
        assert cost_new_way < cost_old_way

        # Savings should be approximately 18% (from journal)
        savings_percent = (cost_old_way - cost_new_way) / cost_old_way * 100
        assert savings_percent > 15  # At least 15% savings
        assert savings_percent < 25  # But not more than 25%

    def test_cache_write_tokens_parameter_accepted(self, calc: CostCalculator) -> None:
        """Cache write tokens parameter should be accepted."""
        # cache_write_tokens is accepted by genai-prices
        # The pricing behavior varies by model - some may charge premium, others not
        cost = calc.calculate_cost(
            "gpt-4o",
            input_tokens=1000,
            output_tokens=0,
            cache_write_tokens=500,
        )
        assert isinstance(cost, int)
        assert cost >= 0

    def test_zero_cache_tokens_same_as_default(self, calc: CostCalculator) -> None:
        """Passing cache_read_tokens=0 should be same as not passing it."""
        cost_default = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)

        cost_explicit_zero = calc.calculate_cost(
            "gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

        assert cost_default == cost_explicit_zero

    def test_backward_compatibility(self, calc: CostCalculator) -> None:
        """Old-style calls without cache args should still work."""
        # This is the old API signature
        cost = calc.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, int)
        assert cost > 0


class TestCostCalculatorAudioTokens:
    """Tests for audio token pricing (multimodal models)."""

    @pytest.fixture
    def calc(self) -> CostCalculator:
        return CostCalculator()

    def test_audio_tokens_accepted(self, calc: CostCalculator) -> None:
        """Should accept audio token parameters without error."""
        # May not affect cost if model doesn't support audio, but shouldn't error
        cost = calc.calculate_cost(
            "gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            input_audio_tokens=100,
            output_audio_tokens=50,
        )
        assert isinstance(cost, int)

    def test_cache_audio_tokens_accepted(self, calc: CostCalculator) -> None:
        """Should accept cache audio token parameters."""
        # cache_audio_read_tokens must not exceed input_audio_tokens or cache_read_tokens
        # genai-prices validates: cache_audio_read_tokens <= cache_read_tokens
        cost = calc.calculate_cost(
            "gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            input_audio_tokens=200,
            cache_read_tokens=150,  # need cache_read >= cache_audio_read
            cache_audio_read_tokens=100,  # 100 of 200 audio tokens cached
        )
        assert isinstance(cost, int)
