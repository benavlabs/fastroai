"""Base classes for organizing AI tools into toolsets.

Toolsets allow grouping related tools together and marking them
with metadata about their safety characteristics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic_ai.toolsets import FunctionToolset


class FunctionToolsetBase(FunctionToolset):
    """Base class for organized tool sets.

    Extends PydanticAI's FunctionToolset with a name for identification
    and organization purposes.

    Examples:
        ```python
        from fastroai.tools import safe_tool, FunctionToolsetBase

        @safe_tool(timeout=30)
        async def web_search(query: str) -> str:
            '''Search the web.'''
            ...

        @safe_tool(timeout=10)
        async def get_weather(location: str) -> str:
            '''Get weather for location.'''
            ...

        class WebToolset(FunctionToolsetBase):
            def __init__(self):
                super().__init__(
                    tools=[web_search, get_weather],
                    name="web",
                )

        # Use with FastroAgent
        agent = FastroAgent(toolsets=[WebToolset()])
        ```
    """

    def __init__(
        self,
        tools: list[Callable[..., Any]],
        name: str | None = None,
    ) -> None:
        """Initialize toolset with tools and optional name.

        Args:
            tools: List of tool functions to include.
            name: Name for this toolset. Defaults to class name.
        """
        super().__init__(tools=tools)
        self.name = name or self.__class__.__name__


class SafeToolset(FunctionToolsetBase):
    """Base class for toolsets containing only safe tools.

    Safe tools are those that:
    - Don't access external networks (or have timeout protection)
    - Don't modify system state
    - Have bounded execution time
    - Return graceful error messages instead of raising exceptions

    Use this as a base class to mark toolsets as production-safe.

    Examples:
        ```python
        @safe_tool(timeout=5)
        async def calculator(expression: str) -> str:
            '''Evaluate a math expression.'''
            try:
                # Safe: no network, no state modification
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        @safe_tool(timeout=1)
        async def get_time() -> str:
            '''Get current time.'''
            from datetime import datetime
            return datetime.now().isoformat()

        class UtilityToolset(SafeToolset):
            def __init__(self):
                super().__init__(
                    tools=[calculator, get_time],
                    name="utilities",
                )
        ```
    """

    pass
