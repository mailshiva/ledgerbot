"""
src/agent/unified_tools.py

Unified tool registry combining credit card, bank, portfolio, and live card tools.

Provides:
  - ALL_TOOL_DEFINITIONS: merged list for LLM function calling
  - execute_any_tool(name, db, **kwargs): single dispatcher for all tools
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
from src.agent.portfolio_tools import (
    PORTFOLIO_TOOL_DEFINITIONS,
    execute_portfolio_tool as _execute_portfolio_tool,
)
from src.agent.live_card_tools import (
    LIVE_CARD_TOOL_DEFINITIONS,
    LIVE_CARD_TOOL_HANDLERS,
)


# ---------------------------------------------------------------------------
# Merged tool definitions
# ---------------------------------------------------------------------------

ALL_TOOL_DEFINITIONS = (
    CREDIT_TOOL_DEFINITIONS
    + BANK_TOOL_DEFINITIONS
    + PORTFOLIO_TOOL_DEFINITIONS
    + LIVE_CARD_TOOL_DEFINITIONS
)

# Build name→source lookup for routing
_CREDIT_TOOL_NAMES = {t["function"]["name"] for t in CREDIT_TOOL_DEFINITIONS}
_BANK_TOOL_NAMES = {t["function"]["name"] for t in BANK_TOOL_DEFINITIONS}
_PORTFOLIO_TOOL_NAMES = {t["function"]["name"] for t in PORTFOLIO_TOOL_DEFINITIONS}
_LIVE_CARD_TOOL_NAMES = {t["function"]["name"] for t in LIVE_CARD_TOOL_DEFINITIONS}


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def execute_any_tool(name: str, db, **kwargs) -> Any:
    """
    Execute a tool by name, routing to the correct handler.

    Args:
        name: Tool function name
        db:   Database manager (DatabaseManager or DualWriteManager)
        **kwargs: Tool-specific arguments

    Returns:
        Tool result (list[dict], dict, or error dict)

    Raises:
        KeyError: If tool name is not registered in any set
    """
    if name in _LIVE_CARD_TOOL_NAMES:
        # Live card tools don't need db — they call Plaid directly
        handler = LIVE_CARD_TOOL_HANDLERS[name]
        return handler(**kwargs)
    if name in _CREDIT_TOOL_NAMES:
        return _execute_credit_tool(name, db, **kwargs)
    if name in _BANK_TOOL_NAMES:
        return _execute_bank_tool(name, db, **kwargs)
    if name in _PORTFOLIO_TOOL_NAMES:
        return _execute_portfolio_tool(name, db, **kwargs)

    all_names = sorted(
        _CREDIT_TOOL_NAMES
        | _BANK_TOOL_NAMES
        | _PORTFOLIO_TOOL_NAMES
        | _LIVE_CARD_TOOL_NAMES
    )
    raise KeyError(f"Unknown tool: {name}. Available: {all_names}")