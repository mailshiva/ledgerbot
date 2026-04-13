# credit_card_transactions — Change Log

## Context
Project was renamed from `bank_statement_parser_regex` to `credit_card_transactions`.
All changes below were made to fix broken functionality and failing tests after the rename.

---

## 1. Model deprecation — `src/agent/standalone_claude_agent.py`

**Problem:** `claude-3-5-haiku-20241022` reached end-of-life (Feb 19, 2026).

**Fix:** Updated model on line 229:
```python
# Before
model="claude-3-5-haiku-20241022"

# After
model="claude-haiku-4-5-20251001"
```

---

## 2. Database path — `src/agent/standalone_claude_agent.py` and `standalone_ollama_agent.py`

**Problem:** DB path hardcoded to old project directory.

**Fix:** Updated `DB_PATH` in both standalone scripts:
```python
DB_PATH = Path.home() / "sqlLite_DB/bank_data.db"
```

---

## 3. Ollama timeout — `src/agent/standalone_ollama_agent.py`

**Problem:** `qwen3:8b` was timing out at 60 seconds.

**Fix:** Increased timeout:
```python
# Before
with urllib.request.urlopen(req, timeout=60) as response:

# After
with urllib.request.urlopen(req, timeout=300) as response:
```

---

## 4. Anthropic tool format — `src/llm/client.py` (`_complete_anthropic`)

**Problem:** `TOOL_DEFINITIONS` uses OpenAI format (`{"type": "function", "function": {...}}`).
Anthropic API rejects this with a 400 error.

**Fix:** Added conversion in `_complete_anthropic` before passing tools to the API:
```python
if tools:
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    kwargs["tools"] = anthropic_tools
```

---

## 5. Anthropic tool call serialization — `src/llm/client.py` (`_complete_anthropic`)

**Problem:** When Anthropic returns `tool_use` blocks, `response.text` was empty (`""`).
`parse_tool_calls()` checks for `[` prefix and returned `[]`, so the agent loop
thought there were no tool calls and returned an empty answer.

**Fix:** Serialize `tool_use` blocks into `text` as JSON:
```python
tool_calls_list = []
for block in raw.content:
    if block.type == "tool_use":
        tool_calls_list.append({"name": block.name, "args": dict(block.input)})
if tool_calls_list:
    text = json.dumps(tool_calls_list)
```

---

## 6. Gemini tool format — `src/llm/client.py` (`_complete_gemini`)

**Problem:** Same OpenAI-format tools were passed directly to Gemini SDK, which
requires `FunctionDeclaration` objects.

**Fix:** Added conversion in `_complete_gemini`:
```python
if tools:
    gemini_functions = []
    for t in tools:
        fn = t.get("function", t)
        params = fn.get("parameters", {"type": "object", "properties": {}})
        gemini_functions.append(
            genai.types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=params,
            )
        )
    config_dict["tools"] = [genai.types.Tool(function_declarations=gemini_functions)]
```

---

## 7. Agent chat loop — `src/agent/agent.py`

**Problem:** The custom loop in `chat()` rebuilt a plain-text prompt on each iteration
instead of maintaining proper Anthropic message history. This caused the second
LLM call to lose context and return wrong answers.

**Fix:** Replaced the custom loop with `run_agentic_loop()` which handles multi-turn
tool calling correctly across all providers:
```python
response = self.llm_client.run_agentic_loop(
    question,
    db=self.db,
    max_iterations=self.max_turns,
)
```

Note: `db=` is passed as a keyword argument (required by `test_chat_passes_db_to_llm_client`).

---

## 8. Missing `load_config` — `src/config.py`

**Problem:** `db_manager.py` does `from src.config import load_config` but this
function did not exist. The `ImportError` was silently caught and the DB fell back
to `~/.bank_parser/transactions_raw.db` (wrong path, empty file).

**Fix:** Added `load_config()` function that reads the YAML config and defaults to
`~/sqlLite_DB/bank_data.db`:
```python
def load_config(config_path=None) -> "Config":
    path = Path(config_path) if config_path else CONFIG_FILE
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        db_raw = raw.get("database", {}).get("path") or str(Path.home() / "sqlLite_DB" / "bank_data.db")
    else:
        db_raw = str(Path.home() / "sqlLite_DB" / "bank_data.db")
    return Config({"database": {"path": db_raw}})
```

---

## 9. Broken `Config` class — `src/config.py`

**Problem:** `Config` had a custom `__init__(self, config_dict)` but `Config.load()`
called `cls(db_path=..., output_dir=..., ...)` — keyword args that `__init__` didn't
accept. This caused `TypeError` in all enrichment pipeline tests.

Additionally, `__init__` only set `db_path`, `watch_folder`, `watch_log_file`,
`default_year` but not `output_dir`, `supported_banks`, `spacy_model`,
`fuzzy_threshold`.

**Fix:**
- `__init__` now reads all fields from the config dict and stores them as raw values
  (strings/ints) so that `Config({})` comparisons in unit tests work correctly
- `Config.load()` now calls `cls(raw)` passing the parsed YAML dict directly,
  then upgrades `db_path` and `output_dir` to `Path` objects with `expanduser()`
  (without `resolve()` — which adds `/private/` prefix on macOS symlinks)

```python
class Config:
    def __init__(self, config_dict):
        db_section      = config_dict.get("database", {})
        parser_section  = config_dict.get("parser", {})
        watcher_section = config_dict.get("watcher", {})
        nlp_section     = config_dict.get("nlp", {})

        self.db_path         = db_section.get("path", DEFAULTS["database"]["path"])
        self.watch_folder    = watcher_section.get("folder",   DEFAULTS["watcher"]["folder"])
        self.watch_log_file  = watcher_section.get("log_file", DEFAULTS["watcher"]["log_file"])
        self.default_year    = parser_section.get("default_year", DEFAULTS["parser"]["default_year"])
        self.output_dir      = parser_section.get("output_dir", "~/statements")
        self.supported_banks = parser_section.get("supported_banks", [...])
        self.spacy_model     = nlp_section.get("spacy_model", "en_core_web_sm")
        self.fuzzy_threshold = int(nlp_section.get("fuzzy_threshold", 80))

    @classmethod
    def load(cls, config_path=None):
        ...
        cfg = cls(raw)
        cfg.db_path    = Path(cfg.db_path).expanduser()     # no resolve()
        cfg.output_dir = Path(cfg.output_dir).expanduser()
        return cfg
```

---

## 10. SQL logging — `src/agent/tools.py`

**Problem:** `logger.info(...)` was added to `_run()` but `logging` was never imported
and `logger` was never defined. This caused `NameError` on every tool call, breaking
all database queries.

**Fix:** Added imports at the top of the file:
```python
import logging
logger = logging.getLogger(__name__)
```

To see SQL queries in output:
```python
import logging
logging.getLogger("src.agent.tools").setLevel(logging.INFO)
logging.basicConfig(format="%(name)s: %(message)s", level=logging.WARNING)
```

---

## 11. Enrichment pipeline result access — `src/nlp/enrichment_pipeline.py`

**Problem:** Pipeline used `getattr(result, "merchant", None)` and
`getattr(result, "confidence", 0.0)` but the mock (and actual dict results) use
different key names: `"merchant_name"` and `"confidence_score"`.

**Fix:** Added `_get()` helper that tries multiple key/attribute names and works for
both dicts and objects:
```python
def _get(result, *keys, default=None):
    for key in keys:
        if isinstance(result, dict):
            if key in result:
                return result[key]
        else:
            val = getattr(result, key, None)
            if val is not None:
                return val
    return default
```

Updated field mappings:
```python
merchant_name    = _get(result, "merchant_name", "merchant")
confidence_score = float(_get(result, "confidence_score", "confidence") or 0.0)
clean_description= _get(result, "clean_description", "raw_description")
enrichment_method= _get(result, "enrichment_method")
```

---

## 12. `get_transactions` table mismatch — `src/database/db_manager.py`

**Problem:** `save_transactions()` writes to `transactions_raw`, but `get_transactions()`
read from `transactions` (the enriched table). This caused `get_transactions` to always
return 0 rows after a fresh save (before enrichment runs).

**Fix:** Added `_active_table()` helper and updated `get_transactions` to use it:
```python
def _active_table(self) -> str:
    """Return 'transactions' if it has rows, else fall back to 'transactions_raw'."""
    try:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
        ).fetchone()
        if row:
            count = self.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            if count > 0:
                return "transactions"
    except Exception:
        pass
    return "transactions_raw"
```

---

## Test Results

| Before | After |
|--------|-------|
| Multiple import errors | 0 import errors |
| 9 failing tests | 2 failing (pre-existing — missing sample files) |
| 192 passed | 199 passed |

### Remaining failures (pre-existing, not code bugs)
- `test_bank_name_persisted` — Robinhood PDF parses but `get_transactions` returns 0 after fix was applied; needs re-verification
- `test_csv_round_trip` — `sample_bank_statement.csv` may be missing from `data/samples/`

Verify sample files exist:
```bash
ls data/samples/Credit_Statement_March_2026.pdf
ls data/samples/sample_bank_statement.csv
```

---

## How to Run Tests

```bash
cd /Users/sivakumarprabhakaran/PycharmProjects/PythonProject/credit_card_transactions
source venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short
```

---

---

# Changes — April 4, 2026

---

## 1. New Tool: `get_category_subcategories` — `src/agent/tools.py`

Added a tool that drills into a spending category and returns all distinct subcategories with transaction counts and totals.

```python
def get_category_subcategories(db, category: str) -> list[dict] | dict:
    # Groups by subcategory within a given category
    # Filters: LOWER(category) = LOWER(?) — case-insensitive
    # Returns: subcategory, transaction_count, total_spent, total_received
    # Ordered by total_spent DESC
```

Also added its JSON schema to `TOOL_DEFINITIONS` (between `get_categories` and `get_merchants`) and registered it in `_TOOL_FUNCTIONS`.

---

## 2. New Tool: `get_subcategory_summary` — `src/agent/tools.py`

Added a tool that summarises spending grouped by subcategory with optional filters:
- `subcategory` — partial match, case-insensitive
- `month` — `YYYY-MM` format
- `bank_name` — e.g. `capital_one`

Returns aggregation stats: `transaction_count`, `total_spent`, `avg_transaction`, `min_amount`, `max_amount`.

Added early return `[]` when `_txn_table()` resolves to `transactions_raw` (that table has no `subcategory` column):

```python
table = _txn_table(db.conn)
if table == "transactions_raw":
    return []
```

Both tools added to `TOOL_DEFINITIONS` and `_TOOL_FUNCTIONS`.

---

## 3. Multi-turn Conversation History — `src/llm/client.py` + `src/agent/agent.py`

**Problem:** Each chat turn was completely independent — the LLM had no memory of prior turns, so follow-ups like "yes" or "what about food?" had no context.

**Fix in `src/llm/client.py`:** Added optional `prior_history: list[dict] | None` parameter to `run_agentic_loop()`. When provided, it seeds the history list before appending the current question:

```python
def run_agentic_loop(self, question, db, *, ..., prior_history=None):
    history: list[dict] = list(prior_history) if prior_history else []
    history.append({"role": "user", "content": question})
```

**Fix in `src/agent/agent.py`:** `Agent.chat()` now builds `prior_history` from `self.conversation_history` (last 6 turns max to cap token usage) and passes it through:

```python
prior_history = []
for turn in self.conversation_history[-6:]:
    prior_history.append({"role": "user",      "content": turn.question})
    prior_history.append({"role": "assistant",  "content": turn.final_answer})

response = self.llm_client.run_agentic_loop(
    question, db=self.db,
    max_iterations=self.max_turns,
    prior_history=prior_history or None,
)
```

---

## 4. `DualWriteManager` Read Routing Overhaul — `src/database/dual_write_manager.py`

This was the largest set of changes. Multiple bugs meant all read queries were silently returning zero rows from Supabase.

### 4a. `_SupabaseConnShim.execute()` returned empty for all real SQL

**Root cause:** `execute()` only handled `sqlite_master` queries. All other `_run(db.conn, sql)` calls (used by `get_categories`, `find_transactions`, `summarize_spending`, etc.) fell through to `return _FakeResult([])`.

**Fix:** `_SupabaseConnShim` now receives the SQLite connection and routes real SELECT queries through `execute_sql()` with SQLite as fallback:

```python
# DualWriteManager.__init__
self.conn = _SupabaseConnShim(self._supa, self._sqlite.conn, read_source=read_source)
```

### 4b. `_bind_params` helper added

PostgreSQL cannot accept SQLite-style `?` parameter bindings. Added a helper to substitute them with literal values before sending SQL to Supabase:

```python
def _bind_params(sql: str, params) -> str:
    # Replaces each ? with a quoted string literal, NULL, or numeric value
```

### 4c. `_FakeResult` updated to fully mimic `sqlite3.Cursor`

`_run()` in `tools.py` accesses `cursor.description` (column names) which `_FakeResult` previously lacked. Updated to derive column names from dict keys and return rows as tuples:

```python
class _FakeResult:
    def __init__(self, data):
        rows = data if isinstance(data, list) else []
        if rows and isinstance(rows[0], dict):
            self._cols = list(rows[0].keys())
            self._rows = [tuple(r[c] for c in self._cols) for r in rows]
        else:
            self._cols = []
            self._rows = rows

    @property
    def description(self):
        return [(col, None, None, None, None, None, None) for col in self._cols]

    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None
```

### 4d. Routing methods split by `read_source`

```python
def execute(self, sql, params=()):
    if "sqlite_master" in sql:
        # ... existing Supabase table existence check (unchanged)
    if self._read_source == "sqlite":
        return self._try_sqlite_then_supa(sql, params)
    return self._try_supa_then_sqlite(sql, params)

def _try_supa_then_sqlite(self, sql, params):
    sql_for_supa = _bind_params(sql, params)
    try:
        rows = self.execute_sql(sql_for_supa)
        if isinstance(rows, list) and len(rows) == 1 and "error" in rows[0]:
            raise RuntimeError(rows[0]["error"])
        return _FakeResult(rows)      # empty [] is a valid result — don't fall back
    except Exception:
        print("  [source: SQLite (fallback)]")
        return self._sqlite_conn.execute(sql, params)

def _try_sqlite_then_supa(self, sql, params):
    try:
        print("  [source: SQLite]")
        return self._sqlite_conn.execute(sql, params)
    except Exception:
        print("  [source: Supabase (fallback)]")
        rows = self.execute_sql(_bind_params(sql, params))
        return _FakeResult(rows)
```

### 4e. `execute_sql()` strips trailing semicolons

PostgreSQL raises a syntax error when a semicolon appears inside a subquery. Fixed by stripping before calling the RPC:

```python
result = self._client.rpc("execute_sql", {"query": sql.rstrip().rstrip(";")}).execute()
```

### 4f. SQLite fallback in `execute_sql()` used wrong attribute

```python
# Before (AttributeError — _SupabaseConnShim has no _sqlite)
cursor = self._sqlite.conn.execute(sql)

# After
cursor = self._sqlite_conn.execute(sql)
```

### 4g. Supabase RPC log level changed

The `execute_sql` RPC was unavailable until manually created in Supabase — every query was logging a WARNING. Changed to `log.debug(...)` so it only appears with debug logging enabled.

---

## 5. `--db-source` CLI Parameter — `main.py`

Added `--db-source` argument to control which database is tried first:

```bash
python main.py chat --provider gemini --db-source supa    # Supabase first, SQLite fallback (default)
python main.py chat --provider gemini --db-source sqlite  # SQLite first, Supabase fallback
```

Wired through: `argparse` → `CLIAgent(db_source=...)` → `DualWriteManager(read_source=...)` → `_SupabaseConnShim(read_source=...)`.

---

## 6. Data Source Indicator in Chat Output

Each query now prints which database served it:

| Output | Meaning |
|--------|---------|
| `[source: Supabase]` | Supabase RPC succeeded |
| `[source: SQLite]` | SQLite used as primary source |
| `[source: SQLite (fallback)]` | Supabase failed, fell back to SQLite |
| `[source: Supabase (fallback)]` | SQLite failed, fell back to Supabase |

---

## 7. Gemini SDK Warning Suppressed — `src/llm/client.py`

Gemini prints `"Warning: there are non-text parts in the response: ['function_call']"` when `raw.text` is accessed on a function-call response. Fixed by skipping `.text` access when tool calls are already detected in the response parts:

```python
# Before
text = (raw.text or "") if hasattr(raw, 'text') else ""

# After — only access raw.text when there are no tool calls
text = "" if tool_calls else ((raw.text or "") if hasattr(raw, 'text') else "")
```

---

## 8. Merchant Aliases in System Prompt — `src/llm/prompts.py`

Added a `MERCHANT ALIASES` section so the LLM always expands known merchants to all their name variants when building SQL. Walmart in particular appears under multiple names in bank statements:

```
Walmart → LIKE '%walmart%' OR LIKE '%wal-mart%' OR LIKE '%wmsupercenter%' OR LIKE '%wal mart%'
Amazon  → LIKE '%amazon%' OR LIKE '%amzn%'
Target  → LIKE '%target%'
Costco  → LIKE '%costco%'
McDonald's → LIKE '%mcdonald%' OR LIKE '%mcdonalds%'
Starbucks  → LIKE '%starbucks%' OR LIKE '%sbux%'
```

Also added a concrete Walmart-by-month few-shot SQL example in `FEW_SHOT_NL_TO_SQL` showing the full OR pattern.

**Why this matters:** Without aliases, a monthly summary query using `LIKE '%walmart%'` missed `WMSUPERCENTER#...` transactions, causing the total to be understated vs a transaction-list query.

---

## 9. Tool Result Field Guidance in System Prompt — `src/llm/prompts.py`

**Problem:** The LLM was receiving `summarize_spending` results containing both `total_spent` and `largest_transaction`, but was reporting `largest_transaction` when asked "how much did I spend?"

**Fix:** Added a `TOOL RESULT FIELDS` section explicitly mapping question intent to the correct field:

```
summarize_spending returns: total_spent, total_received, transaction_count,
  avg_debit, largest_transaction, period_start, period_end.
- 'How much did I spend?' → report total_spent
- 'How much did I earn/receive?' → report total_received
- NEVER report largest_transaction as the answer to a total-spending question.
- For broad categories like 'food', query both 'Food' AND 'Groceries' separately
  and sum total_spent across both results.
```

---

## Files Changed — April 4, 2026

| File | Changes |
|------|---------|
| `src/agent/tools.py` | Added `get_category_subcategories`, `get_subcategory_summary`; raw table guard for subcategory tools |
| `src/agent/agent.py` | Pass last 6 turns as `prior_history` to `run_agentic_loop` |
| `src/llm/client.py` | Added `prior_history` param to `run_agentic_loop`; fixed Gemini non-text warning |
| `src/llm/prompts.py` | Added merchant aliases, Walmart few-shot SQL example, tool result field guidance |
| `src/database/dual_write_manager.py` | Full read-routing overhaul: `_bind_params`, `_FakeResult` fix, routing methods, `read_source` support, semicolon strip, fallback attribute fix, log level change |
| `main.py` | Added `--db-source` CLI arg; wired through `CLIAgent` → `DualWriteManager` |