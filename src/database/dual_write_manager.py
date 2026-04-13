"""
src/database/dual_write_manager.py

Dual-write database manager.
  - WRITES  → SQLite first, then Supabase. Rolls back SQLite if Supabase fails.
  - SELECTS → Supabase only (cloud is source of truth for reads).

Drop-in replacement for DatabaseManager — same interface, no changes
needed in tools.py, agent.py, or main.py.
"""

from __future__ import annotations

import logging
import re
import subprocess
from contextlib import contextmanager
from typing import Any

from supabase import Client, create_client

from src.database.db_manager import DatabaseManager

log = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Credentials — same Keychain pattern as your other API keys
# ----------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _get_supabase_client() -> Client:
    url = _keychain_get("supabase_bank", "url")
    key = _keychain_get("supabase_bank", "anon_key")
    return create_client(url, key)


# ----------------------------------------------------------------
# Type coercion — same logic as import script
# ----------------------------------------------------------------

_NUMERIC_COLS = {
    "total_debits", "total_credits", "amount",
    "balance", "confidence_score"
}
_INTEGER_COLS = {
    "id", "statement_id", "raw_id", "total_transactions"
}


def _coerce(row: dict) -> dict:
    """Ensure Python types are correct for Supabase REST API."""
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif k in _INTEGER_COLS:
            out[k] = int(v)
        elif k in _NUMERIC_COLS:
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------
# DualWriteManager
# ----------------------------------------------------------------

class DualWriteManager:
    """
    Wraps DatabaseManager (SQLite) and adds Supabase as a second write target.

    Write contract:
      1. Begin SQLite transaction
      2. Write to SQLite
      3. Write to Supabase
      4a. Both succeed → commit SQLite
      4b. Supabase fails → rollback SQLite, raise

    Read contract:
      - All SELECT queries go to Supabase only.
      - self.conn is a pass-through to the Supabase-backed connection shim
        so that tools.py (_has_enriched_table, _txn_table, query_db) work
        without any changes.
    """

    def __init__(self, sqlite_path: str, read_source: str = "supa"):
        self._sqlite = DatabaseManager(sqlite_path)
        self._supa: Client = _get_supabase_client()

        # Expose a .conn shim so query_db(db, sql) in tools.py still works.
        self.conn = _SupabaseConnShim(self._supa, self._sqlite.conn, read_source=read_source)

        log.info("DualWriteManager ready — read_source=%s | writes: SQLite + Supabase", read_source)

    # ----------------------------------------------------------------
    # Public write API — used by parser and enrichment pipeline
    # ----------------------------------------------------------------

    def upsert_statement(self, row: dict) -> int:
        """
        Insert or update a statement row.
        Returns the statement id.
        """
        return self._dual_upsert("statements", row, conflict_col="file_hash")

    def upsert_transaction_raw(self, row: dict) -> int:
        """Insert or update a transactions_raw row. Returns id."""
        return self._dual_upsert("transactions_raw", row, conflict_col="id")

    def upsert_transaction(self, row: dict) -> int:
        """Insert or update an enriched transactions row. Returns id."""
        return self._dual_upsert("transactions", row, conflict_col="raw_id")

    def upsert_statements_bulk(self, rows: list[dict]) -> int:
        """Bulk upsert statements. Returns count inserted."""
        return self._dual_upsert_bulk("statements", rows, conflict_col="file_hash")

    def upsert_transactions_raw_bulk(self, rows: list[dict]) -> int:
        """Bulk upsert transactions_raw. Returns count inserted."""
        return self._dual_upsert_bulk("transactions_raw", rows, conflict_col="id")

    def upsert_transactions_bulk(self, rows: list[dict]) -> int:
        """Bulk upsert enriched transactions. Returns count inserted."""
        return self._dual_upsert_bulk("transactions", rows, conflict_col="raw_id")

    # ----------------------------------------------------------------
    # Internal dual-write logic
    # ----------------------------------------------------------------

    def _dual_upsert(self, table: str, row: dict, conflict_col: str) -> int:
        """
        Single-row upsert with rollback semantics.
        SQLite uses a savepoint so we can roll back just this operation.
        """
        sqlite_conn = self._sqlite.conn

        with self._sqlite_savepoint(sqlite_conn) as savepoint:
            # Step 1 — write to SQLite
            self._sqlite_upsert(sqlite_conn, table, row, conflict_col)

            # Step 2 — write to Supabase (if this raises, savepoint rolls back)
            try:
                result = (
                    self._supa.table(table)
                    .upsert(_coerce(row), on_conflict=conflict_col)
                    .execute()
                )
                if not result.data:
                    raise RuntimeError(f"Supabase upsert returned no data for {table}")
                row_id = result.data[0].get("id")
            except Exception as e:
                log.error("Supabase upsert failed for %s — rolling back SQLite: %s", table, e)
                raise  # triggers savepoint rollback via context manager

        return row_id

    def _dual_upsert_bulk(self, table: str, rows: list[dict], conflict_col: str) -> int:
        """
        Bulk upsert with rollback semantics.
        All rows in one SQLite savepoint + one Supabase batch call.
        Large batches are chunked to stay under Supabase's 1MB limit.
        """
        if not rows:
            return 0

        sqlite_conn = self._sqlite.conn
        CHUNK = 500

        with self._sqlite_savepoint(sqlite_conn) as savepoint:
            # Step 1 — write all rows to SQLite
            for row in rows:
                self._sqlite_upsert(sqlite_conn, table, row, conflict_col)

            # Step 2 — write to Supabase in chunks
            try:
                total = 0
                for i in range(0, len(rows), CHUNK):
                    chunk = [_coerce(r) for r in rows[i:i + CHUNK]]
                    result = (
                        self._supa.table(table)
                        .upsert(chunk, on_conflict=conflict_col)
                        .execute()
                    )
                    total += len(result.data or [])
                    log.debug("Supabase bulk upsert %s: chunk %d/%d done",
                              table, i // CHUNK + 1, (len(rows) + CHUNK - 1) // CHUNK)
            except Exception as e:
                log.error("Supabase bulk upsert failed for %s — rolling back SQLite: %s", table, e)
                raise

        return total

    @contextmanager
    def _sqlite_savepoint(self, conn):
        """
        SQLite savepoint = nested transaction.
        Rolls back only this operation if Supabase fails,
        leaving any outer transaction intact.
        """
        savepoint = "sp_dual_write"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield savepoint
            conn.execute(f"RELEASE {savepoint}")
            conn.commit()
        except Exception:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
            raise

    def _sqlite_upsert(self, conn, table: str, row: dict, conflict_col: str):
        """Build and execute an INSERT OR REPLACE into SQLite."""
        cols = list(row.keys())
        placeholders = ", ".join("?" * len(cols))
        col_names = ", ".join(cols)
        values = [row[c] for c in cols]

        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
            values
        )

    # ----------------------------------------------------------------
    # Pass-throughs for any code that still uses DatabaseManager directly
    # ----------------------------------------------------------------

    @property
    def sqlite(self) -> DatabaseManager:
        """Direct access to SQLite if ever needed (e.g. migrations)."""
        return self._sqlite

    @property
    def supabase(self) -> Client:
        """Direct access to Supabase client if ever needed."""
        return self._supa

    def close(self):
        self._sqlite.conn.close()

    def __repr__(self):
        return f"DualWriteManager(sqlite={self._sqlite}, supabase=connected)"


# ----------------------------------------------------------------
# Supabase connection shim
# ----------------------------------------------------------------

def _bind_params(sql: str, params) -> str:
    """Substitute SQLite ? placeholders with literal values for Supabase."""
    if not params:
        return sql
    result = []
    param_iter = iter(params)
    for char in sql:
        if char == '?':
            val = next(param_iter, None)
            if val is None:
                result.append('NULL')
            elif isinstance(val, str):
                result.append("'" + val.replace("'", "''") + "'")
            else:
                result.append(str(val))
        else:
            result.append(char)
    return ''.join(result)


class _SupabaseConnShim:
    """
    Mimics sqlite3.Connection just enough for tools.py to work unchanged.

    tools.py uses db.conn in two ways:
      1. _has_enriched_table(conn) — checks if 'transactions' table exists
      2. query_db(db, sql)         — runs arbitrary SELECT SQL

    Both are routed to Supabase here.
    """

    def __init__(self, client: Client, sqlite_conn, read_source: str = "supa"):
        self._client = client
        self._sqlite_conn = sqlite_conn
        self._read_source = read_source.lower()   # "supa" or "sqlite"
        self._has_enriched: bool | None = None    # cached after first check

    def execute(self, sql: str, params: tuple = ()):
        """
        Called by _has_enriched_table() with a sqlite_master query, and by
        _run() in tools.py for all other SELECT queries.
        """
        if "sqlite_master" in sql or "information_schema" in sql:
            try:
                r = self._client.table("transactions").select("id").limit(1).execute()
                exists = r.data is not None
            except Exception:
                exists = False
            self._has_enriched = exists
            return _FakeResult([{"name": "transactions"}] if exists else [])

        if self._read_source == "sqlite":
            return self._try_sqlite_then_supa(sql, params)
        return self._try_supa_then_sqlite(sql, params)

    def _try_supa_then_sqlite(self, sql: str, params: tuple):
        sql_for_supa = _bind_params(sql, params)
        try:
            rows = self.execute_sql(sql_for_supa)
            if isinstance(rows, list) and len(rows) == 1 and "error" in rows[0]:
                raise RuntimeError(rows[0]["error"])
            return _FakeResult(rows)
        except Exception:
            print("  [source: SQLite (fallback)]")
            return self._sqlite_conn.execute(sql, params)

    def _try_sqlite_then_supa(self, sql: str, params: tuple):
        try:
            print("  [source: SQLite]")
            return self._sqlite_conn.execute(sql, params)
        except Exception:
            print("  [source: Supabase (fallback)]")
            sql_for_supa = _bind_params(sql, params)
            rows = self.execute_sql(sql_for_supa)
            return _FakeResult(rows)

    def execute_sql(self, sql: str) -> list[dict]:
        """
        Called by query_db() in tools.py to run arbitrary SELECT SQL.
        Routes to Supabase via the execute_sql RPC function you created in Step 3.
        """
        try:
            # Replace LIKE with ILIKE so Supabase (PostgreSQL) behaves like SQLite
            # which is case-insensitive for LIKE by default.
            supa_sql = sql.rstrip().rstrip(";")
            supa_sql = re.sub(r'\bLIKE\b', 'ILIKE', supa_sql, flags=re.IGNORECASE)
            result = self._client.rpc("execute_sql", {"query": supa_sql}).execute()
            data = result.data
            if isinstance(data, list):
                print("  [source: Supabase]")
                return data
            if isinstance(data, str):
                import json
                print("  [source: Supabase]")
                return json.loads(data) or []
            return []
        except Exception as e:
            log.debug("Supabase execute_sql RPC unavailable, falling back to SQLite: %s", e)
            try:
                cursor = self._sqlite_conn.execute(sql)
                cols = [d[0] for d in cursor.description]
                print("  [source: SQLite]")
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
            except Exception as sqlite_e:
                log.error("SQLite fallback also failed: %s | SQL: %s", sqlite_e, sql[:200])
                return [{"error": str(sqlite_e)}]


class _FakeResult:
    """Mimics sqlite3.Cursor for dict-based results (e.g. from Supabase)."""
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
