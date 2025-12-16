"""Decorators for production-safe AI tools.

This module provides the @safe_tool decorator that adds timeout, retry,
and error handling to AI tools. When a tool fails, it returns an error
message instead of raising an exception, allowing the AI to handle
the failure gracefully.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

logger = logging.getLogger("fastroai.tools")

DEFAULT_TOOL_TIMEOUT = 30
DEFAULT_TOOL_MAX_RETRIES = 3

P = ParamSpec("P")
R = TypeVar("R")


def safe_tool(
    timeout: float = DEFAULT_TOOL_TIMEOUT,
    max_retries: int = DEFAULT_TOOL_MAX_RETRIES,
    on_timeout: str | None = None,
    on_error: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R | str]]]:
    """Decorator that adds timeout, retry, and error handling to AI tools.

    When a tool times out or raises an exception, instead of crashing the
    conversation, this decorator returns an error message that the AI can
    use to respond gracefully.

    Args:
        timeout: Maximum seconds per attempt. Default: 30.
        max_retries: Maximum retry attempts. Default: 3.
        on_timeout: Custom message returned on timeout.
                   Default: "Tool timed out after {max_retries} attempts"
        on_error: Custom message returned on error.
                 Use {error} placeholder for error details.
                 Default: "Tool failed: {error}"

    Returns:
        Decorated async function with safety features.

    Examples:
        ```python
        @safe_tool(timeout=10, max_retries=2)
        async def web_search(query: str) -> str:
            '''Search the web for information.'''
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://api.example.com?q={query}")
                return response.text

        # If the API is slow or down:
        # - Waits max 10 seconds per attempt
        # - Retries up to 2 times with exponential backoff
        # - Returns error message on final failure
        # - AI sees: "Tool timed out after 2 attempts"

        # With custom messages:
        @safe_tool(
            timeout=30,
            on_timeout="Search is taking too long. Try a simpler query.",
            on_error="Search unavailable: {error}",
        )
        async def search(query: str) -> str:
            ...
        ```
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R | str]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | str:
            last_error: Exception | None = None
            func_name = func.__name__

            for attempt in range(max_retries):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout,
                    )
                except TimeoutError:
                    logger.warning(f"Tool '{func_name}' timeout on attempt {attempt + 1}/{max_retries}")
                    last_error = TimeoutError(f"Timeout after {timeout}s")
                except Exception as e:
                    logger.warning(f"Tool '{func_name}' failed on attempt {attempt + 1}/{max_retries}: {e}")
                    last_error = e

                if attempt < max_retries - 1:
                    backoff = 2**attempt * 0.1
                    await asyncio.sleep(backoff)

            if isinstance(last_error, asyncio.TimeoutError):
                return on_timeout or f"Tool timed out after {max_retries} attempts"

            error_msg = str(last_error) if last_error else "Unknown error"
            if on_error:
                return on_error.format(error=error_msg)
            return f"Tool failed: {error_msg}"

        return wrapper

    return decorator
