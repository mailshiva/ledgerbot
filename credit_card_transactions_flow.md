# Credit Card Transactions — Control Flow Diagram

Traces the execution path from `main.py` through every component touched at runtime.

---

## High-Level Flow

```mermaid
flowchart TD
    CLI["<b>main.py</b>\nmain()"]
    CLIAGENT["<b>main.py</b>\nCLIAgent.__init__()"]
    CONFIG["<b>src/llm/config.py</b>\nLLMConfig\n─────────────────\n• Resolves API keys\n  (macOS Keychain → env var)\n• Holds model defaults\n• Cost table per model\n• validate() checks key present"]
    CLIENT["<b>src/llm/client.py</b>\nLLMClient\n─────────────────\n• complete_with_tools()\n• parse_tool_calls()\n• In-process response cache\n• UsageSummary tracking"]
    DWM["<b>src/database/dual_write_manager.py</b>\nDualWriteManager\n─────────────────\n• Wraps DatabaseManager (SQLite)\n• Wraps Supabase client\n• Reads: Supabase → SQLite fallback\n• Writes: SQLite savepoint + Supabase"]
    GRAPH["<b>src/agent/graph.py</b>\nGraphAgent\n─────────────────\n• chat() / invoke()\n• Builds AgentState\n• Runs _run_graph() loop\n• Records Turn history"]
    CHAT["<b>main.py</b>\nCLIAgent.chat(question)"]
    PROMPT["<b>src/llm/prompts.py</b>\nAGENT_SYSTEM_PROMPT\n─────────────────\n• DB schema description\n• Tool usage instructions\n• Finance domain rules"]

    CLI -->|"parse args\ncreate CLIAgent"| CLIAGENT
    CLIAGENT -->|"LLMConfig(model)"| CONFIG
    CONFIG -->|"config validated"| CLIENT
    CLIAGENT -->|"DualWriteManager(db_path, read_source)"| DWM
    CLIAGENT -->|"GraphAgent(db, llm_client)"| GRAPH
    GRAPH -->|"loads"| PROMPT
    CLI -->|"run_interactive() loop\nor piped stdin"| CHAT
    CHAT -->|"agent.chat(question)"| GRAPH
```

---

## GraphAgent Internal Loop

```mermaid
flowchart TD
    START(["START\ninvoke(question)"])
    INIT["Build AgentState\n─────────────────\n• Prepend last 6 turns\n  as prior_history\n• messages = history + question\n• iteration = 0"]
    LLM["<b>Node: call_llm</b>\n─────────────────\n• Flatten messages → prompt string\n• llm_client.complete_with_tools(\n    prompt,\n    ALL_TOOL_DEFINITIONS,\n    system=AGENT_SYSTEM_PROMPT\n  )\n• parse_tool_calls(response)\n• Accumulate tokens + cost\n• iteration += 1"]
    EDGE{"<b>Edge: should_continue</b>\n─────────────────\ntool_calls present\nAND iteration < max_iterations?"}
    TOOLS["<b>Node: call_tools</b>\n─────────────────\n• For each tool call:\n    execute_any_tool(name, db, **args)\n• Append assistant turn to messages\n• Append tool results to messages\n• Clear tool_calls for next loop"]
    SUMMARY["<b>Node: force_summary</b>\n─────────────────\n• If max_iterations reached:\n    inject 'summarise what you have'\n    llm_client.complete() once more\n• Else: use last LLM text as answer"]
    END(["END\nReturn final_answer\nRecord Turn"])

    START --> INIT --> LLM --> EDGE
    EDGE -->|"YES — tools needed"| TOOLS
    TOOLS -->|"loop back"| LLM
    EDGE -->|"NO — plain answer\nor max iters hit"| SUMMARY
    SUMMARY --> END
```

---

## LLMClient Provider Dispatch

```mermaid
flowchart LR
    CWT["LLMClient\n.complete_with_tools()"]
    LOOKUP["MODEL_PROVIDER lookup\n─────────────────\nModel → Provider"]
    ADAPTER{"Provider?"}
    GEMINI["_complete_gemini()\ngoogle.genai SDK\nFunctionDeclaration tools"]
    ANTHROPIC["_complete_anthropic()\nanthropic SDK\ntool_use blocks"]
    OPENAI["_complete_openai()\nopenai SDK\nOpenAI-style tools"]
    OLLAMA["_complete_ollama()\nurllib POST\n/api/chat\n(no extra deps)"]
    RESP["LLMResponse\n─────────────────\n• text (or JSON tool calls)\n• input_tokens / output_tokens\n• cost_usd\n• latency_ms"]

    CWT --> LOOKUP --> ADAPTER
    ADAPTER -->|GEMINI| GEMINI
    ADAPTER -->|ANTHROPIC| ANTHROPIC
    ADAPTER -->|OPENAI| OPENAI
    ADAPTER -->|OLLAMA| OLLAMA
    GEMINI & ANTHROPIC & OPENAI & OLLAMA --> RESP
```

---

## Unified Tool Dispatch

```mermaid
flowchart LR
    EAT["execute_any_tool(name, db, **kwargs)"]
    ROUTE{"name in\nwhich set?"}
    CC["_execute_credit_tool()\n─────────────────\nsrc/agent/tools.py\n• query_db\n• get_categories\n• get_merchants\n• summarize_spending"]
    BANK["_execute_bank_tool()\n─────────────────\nsrc/agent/bank_tools.py\n• get_bank_balance\n• (other bank queries)"]
    DB["DualWriteManager\n─────────────────\n_SupabaseConnShim.execute_sql()\n→ Supabase RPC execute_sql\n→ SQLite fallback"]

    EAT --> ROUTE
    ROUTE -->|credit tool names| CC
    ROUTE -->|bank tool names| BANK
    CC & BANK -->|"db.conn.execute_sql(sql)"| DB
```

---

## DualWriteManager Read Path

```mermaid
flowchart TD
    QUERY["Tool calls\nquery_db(sql)"]
    SHIM["_SupabaseConnShim.execute(sql)\n─────────────────\nread_source = 'supa' or 'sqlite'"]
    SUPA_PATH["Supabase RPC\nexecute_sql(query)\n─────────────────\nSQL → ILIKE rewrite\nRPC call → JSON rows"]
    SQLITE_FB["SQLite fallback\n_sqlite_conn.execute(sql)\n─────────────────\nreturns row dicts"]
    RESULT["_FakeResult\nmimics sqlite3.Cursor\n.fetchall() / .description"]

    QUERY --> SHIM
    SHIM -->|"read_source=supa (default)"| SUPA_PATH
    SUPA_PATH -->|"network error / RPC unavailable"| SQLITE_FB
    SHIM -->|"read_source=sqlite"| SQLITE_FB
    SQLITE_FB -->|"exception"| SUPA_PATH
    SUPA_PATH --> RESULT
    SQLITE_FB --> RESULT
```

---

## Component Summary Table

| Component | File | Role |
|-----------|------|------|
| `main()` | `main.py` | CLI entry point; parses `--provider`, `--model`, `--db-source` |
| `CLIAgent` | `main.py` | Session wrapper; tracks turns, cost, pretty-prints |
| `LLMConfig` | `src/llm/config.py` | API key resolution (Keychain → env); model/cost metadata |
| `LLMClient` | `src/llm/client.py` | Unified LLM API; provider dispatch; tool-call parsing; cache |
| `GraphAgent` | `src/agent/graph.py` | Stateful agentic loop (call_llm → should_continue → call_tools) |
| `AGENT_SYSTEM_PROMPT` | `src/llm/prompts.py` | Finance-domain instructions + DB schema injected into every LLM call |
| `execute_any_tool` | `src/agent/unified_tools.py` | Routes tool name → credit tools or bank tools |
| `DualWriteManager` | `src/database/dual_write_manager.py` | Dual read/write: Supabase primary, SQLite fallback |
| `_SupabaseConnShim` | `src/database/dual_write_manager.py` | sqlite3.Connection-compatible shim; translates SQL → Supabase RPC |
| `DatabaseManager` | `src/database/db_manager.py` | SQLite connection; underlying local store |

---

## Key Data Types Flowing Between Components

```
CLIAgent.chat(question: str)
    → GraphAgent.invoke(question: str)
        → AgentState (TypedDict)
            { messages: list[dict],     # role/content pairs
              tool_calls: list[dict],   # [{"name":..., "args":{...}}]
              llm_response: dict,       # text, tokens, cost
              iteration: int,
              final_answer: str }
        → LLMResponse (dataclass)
            { text: str,               # plain text OR JSON tool calls
              input_tokens: int,
              output_tokens: int,
              cost_usd: float }
        → Turn (dataclass)
            { question, final_answer,
              input_tokens, output_tokens,
              cost_usd, timestamp }
    ← str (final_answer)
```