#!/usr/bin/env python3
"""
Standalone Claude/Anthropic Agent for Bank Statement Queries

Run this independently without touching the main codebase.
Works with Claude Haiku, Sonnet, or Opus via Anthropic API.

Usage:
    python standalone_claude_agent.py "How much did I spend on food?"
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Keychain helper (same as config.py)
# ---------------------------------------------------------------------------

def _keychain_get(account: str) -> str:
    """Read a secret from the macOS Keychain."""
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-s", "portfolio_advisor",
                "-a", account,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""

def _get_anthropic_key() -> str:
    """Get Anthropic API key from Keychain or environment."""
    return _keychain_get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")

# Initialize Anthropic client
api_key = _get_anthropic_key()
if not api_key:
    print("❌ Error: No Anthropic API key found!")
    print("\nAdd your key to macOS Keychain:")
    print('  security add-generic-password -a "anthropic_api_key" -s "portfolio_advisor" -w "YOUR_KEY"')
    print("\nOr set environment variable:")
    print('  export ANTHROPIC_API_KEY="your-key"')
    sys.exit(1)

client = Anthropic(api_key=api_key)

# Load database
import sqlite3
DB_PATH = Path.home() / "sqlLite_DB/bank_data.db"
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

SYSTEM_PROMPT = """You are a helpful financial assistant that answers questions about bank transactions.
You have access to a database of transactions with their amounts, dates, merchants, and categories.

When answering questions, first query the database to get actual transaction data, then synthesize a clear answer.
Always provide specific numbers from the data when answering."""

TOOLS = [
    {
        "name": "query_transactions",
        "description": "Query the transactions database with SQL",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute against transactions table"
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "get_categories",
        "description": "Get list of spending categories and amounts",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_merchants",
        "description": "Get list of merchants and spending amounts",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of merchants to return (default 10)"
                }
            }
        }
    }
]

def query_transactions(sql: str) -> list:
    """Execute SQL query on transactions table"""
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_categories() -> list:
    """Get spending by category"""
    try:
        sql = """
        SELECT category, 
               SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as total_spent,
               SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END) as total_received,
               COUNT(*) as transaction_count
        FROM transactions
        WHERE category != 'Uncategorized'
        GROUP BY category
        ORDER BY total_spent DESC
        """
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_merchants(limit: int = 10) -> list:
    """Get spending by merchant"""
    try:
        sql = f"""
        SELECT merchant_name,
               SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as total_spent,
               COUNT(*) as transaction_count
        FROM transactions
        WHERE merchant_name IS NOT NULL
        GROUP BY merchant_name
        ORDER BY total_spent DESC
        LIMIT {limit}
        """
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def process_tool_call(tool_name: str, tool_input: dict):
    """Execute a tool and return results"""
    if tool_name == "query_transactions":
        return query_transactions(tool_input.get("sql", "SELECT * FROM transactions LIMIT 5"))
    elif tool_name == "get_categories":
        return get_categories()
    elif tool_name == "get_merchants":
        return get_merchants(tool_input.get("limit", 10))
    else:
        return [{"error": f"Unknown tool: {tool_name}"}]

def chat_with_tools(user_message: str, conversation_history: list) -> tuple[str, list]:
    """
    Send message to Claude with tools, process tool calls, and return final answer.
    
    Returns:
        (final_answer: str, tool_calls_made: list of tool names)
    """
    
    tools_called = []
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # First API call with tools
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=conversation_history
    )
    
    # Process response
    while response.stop_reason == "tool_use":
        # Find tool use blocks in response
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tools_called.append(tool_name)
                
                print(f"  🔧 Using tool: {tool_name}")
                
                # Execute tool
                tool_result = process_tool_call(tool_name, tool_input)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": json.dumps(tool_result)
                })
        
        # Add assistant response and tool results to history
        conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        conversation_history.append({
            "role": "user",
            "content": tool_results
        })
        
        # Second API call with tool results
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation_history
        )
    
    # Extract final text response
    final_answer = ""
    for content_block in response.content:
        if hasattr(content_block, "text"):
            final_answer += content_block.text
    
    # Add final response to history
    conversation_history.append({
        "role": "assistant",
        "content": final_answer
    })
    
    return final_answer, tools_called

def main():
    """Interactive chat loop"""
    conversation_history = []
    
    print("=" * 80)
    print("CLAUDE BANK STATEMENT AGENT (Standalone)")
    print("=" * 80)
    print("\nAsk questions about your spending:")
    print("  'How much did I spend on food?'")
    print("  'What are my top merchants?'")
    print("  'Show me my spending by category'")
    print("\nType 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            print("\nAgent: ", end="", flush=True)
            answer, tools = chat_with_tools(user_input, conversation_history)
            print(answer)
            
            if tools:
                print(f"\n  Tools used: {', '.join(tools)}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        print(f"\nQuestion: {question}\n")
        answer, tools = chat_with_tools(question, [])
        print(f"Answer: {answer}")
        if tools:
            print(f"Tools used: {', '.join(tools)}")
    else:
        # Interactive mode
        main()
