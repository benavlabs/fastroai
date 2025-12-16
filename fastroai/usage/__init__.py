"""Usage tracking module for cost calculation.

Provides precise cost calculation for AI model usage using integer
microcents to avoid floating-point precision errors in billing.

Uses genai-prices for up-to-date model pricing with support for
custom pricing overrides.
"""

from .calculator import CostCalculator

__all__ = [
    "CostCalculator",
]
