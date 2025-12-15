"""Agent module for AI interactions.

Provides FastroAgent, a PydanticAI wrapper with usage tracking,
cost calculation, and distributed tracing support.
"""

from .agent import AgentStepWrapper, FastroAgent
from .schemas import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    AgentConfig,
    ChatResponse,
    StreamChunk,
)

__all__ = [
    "FastroAgent",
    "AgentStepWrapper",
    "AgentConfig",
    "ChatResponse",
    "StreamChunk",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_SYSTEM_PROMPT",
]
