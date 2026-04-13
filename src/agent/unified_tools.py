"""
src/agent/unified_tools.py

Unified tool registry combining credit card tools and bank tools.

Provides:
  - ALL_TOOL_DEFINITIONS: merged list for LLM function calling
  - execute_any_tool(name, db, **kwargs): single dispatcher for all tools

This module replaces the need to import from both tools.py and bank_tools.py
separately. The agent/graph just calls execute_any_tool() and it routes to
the correct handler.
"""

from __future__ import annotations

from typing import Any

from src.agent.tools import (
    TOOL_DEFINITIONS as CREDIT_TOOL_DEFINITIONS,
    execute_tool as _execute_credit_tool,
)
from src.agent.bank_tools import (
    BANK_TOOL_DEFINITIONS,
    execute_bank_tool as _execute_bank_tool,
)


# ---------------------------------------------------------------------------
# Merged tool definitions
# ---------------------------------------------------------------------------

ALL_TOOL_DEFINITIONS = CREDIT_TOOL_DEFINITIONS + BANK_TOOL_DEFINITIONS

# Build name→source lookup for routing
_CREDIT_TOOL_NAMES = {
    t["function"]["name"] for t in CREDIT_TOOL_DEFINITIONS
}
_BANK_TOOL_NAMES = {
    t["function"]["name"] for t in BANK_TOOL_DEFINITIONS
}


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def execute_any_tool(name: str, db, **kwargs) -> Any:
    """
    Execute a tool by name, routing to either credit card or bank handler.

    Args:
        name: Tool function name (e.g. 'get_categories', 'get_bank_balance')
        db: Database manager (DatabaseManager or DualWriteManager)
        **kwargs: Tool-specific arguments

    Returns:
        Tool result (list[dict], dict, or error dict)

    Raises:
        KeyError: If tool name is not registered in either set
    """
    if name in _CREDIT_TOOL_NAMES:
        return _execute_credit_tool(name, db, **kwargs)
    if name in _BANK_TOOL_NAMES:
        return _execute_bank_tool(name, db, **kwargs)
    raise KeyError(
        f"Unknown tool: {name}. "
        f"Available: {sorted(_CREDIT_TOOL_NAMES | _BANK_TOOL_NAMES)}"
    )