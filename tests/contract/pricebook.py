"""Whole-dollar prices for scripted models, so every contract cost is an exact integer of microcents."""

import dataclasses
from dataclasses import dataclass

from pydantic_ai.usage import RequestUsage, RunUsage

from fastroai import CostCalculator


@dataclass(frozen=True)
class Price:
    input: int
    output: int
    cache_read: int
    cache_write: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            rate = getattr(self, field.name)
            if type(rate) is not int or rate <= 0:
                raise ValueError(
                    f"the {field.name} price must be a positive whole number of dollars per Mtok, got {rate!r}"
                )


PRICES: dict[str, Price] = {
    "scripted-small": Price(input=3, output=15, cache_read=1, cache_write=4),
    "scripted-large": Price(input=5, output=25, cache_read=2, cache_write=7),
}


def calculator() -> CostCalculator:
    return CostCalculator(
        pricing_overrides={
            model: {
                "input_per_mtok": float(price.input),
                "output_per_mtok": float(price.output),
                "cache_read_per_mtok": float(price.cache_read),
                "cache_write_per_mtok": float(price.cache_write),
            }
            for model, price in PRICES.items()
        }
    )


def expected_microcents(model: str, usage: RequestUsage | RunUsage) -> int:
    if model not in PRICES:
        raise ValueError(f"no pricebook price for {model!r}")
    if usage.input_audio_tokens or usage.output_audio_tokens or usage.cache_audio_read_tokens:
        raise ValueError("the pricebook has no audio prices")
    cached = usage.cache_read_tokens + usage.cache_write_tokens
    if cached > usage.input_tokens:
        raise ValueError(f"cache tokens {cached} exceed input tokens {usage.input_tokens}")
    price = PRICES[model]
    return (
        (usage.input_tokens - cached) * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    )
