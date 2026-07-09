"""
Database Manager - Week 1
Handles all SQLite operations. DB path is read from ~/.bank_parser/config.yaml.
Duplicate file detection via SHA-256 hash before any write.

Dual-write behaviour (Week 5):
  save_statement()   — writes to SQLite then Supabase atomically
  save_transactions() — writes to SQLite then Supabase atomically
  All reads (get_transactions, get_statements, etc.) remain SQLite-only
  since they are internal/legacy reads. Agent queries go via Supabase
  through DualWriteManager.
"""

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class DuplicateStatementError(Exception):
    """Raised when a file with the same SHA-256 hash already exists in the DB."""
    def __init__(self, filepath: str, original_path: str, import_date: str):
        self.filepath      = filepath
        self.original_path = original_path
        self.import_date   = import_date
        super().__init__(
            f"File already imported.\n"
            f"  Current file  : {filepath}\n"
            f"  First imported: {original_path}\n"
            f"  Import date   : {import_date}"
        )


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        # Resolve DB path: argument > config > fallback default
        if db_path is None:
            try:
                from src.config import load_config
                cfg = load_config()
                db_path = cfg.db_path
            except Exception:
                db_path = str(Path.home() / ".bank_parser" / "transactions_raw.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._initialize_schema()

        # ── Dual-write: lazy-load Supabase client ─────────────────────
        # Kept optional so unit tests and offline runs work without
        # Supabase credentials. Set to None until first write is attempted.
        self._supa = None

    # ------------------------------------------------------------------ #
    #  Connection & Schema                                                 #
    # ------------------------------------------------------------------ #

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _initialize_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            self.conn.executescript(schema_path.read_text())
        else:
            # Inline fallback
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    bank_name TEXT,
                    account_number TEXT,
                    statement_date TEXT,
                    import_date TEXT DEFAULT (datetime('now')),
                    total_transactions INTEGER DEFAULT 0,
                    total_debits REAL DEFAULT 0.0,
                    total_credits REAL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS transactions_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement_id INTEGER,
                    bank_name TEXT,
                    date TEXT NOT NULL,
                    description TEXT NOT NULL,
                    amount REAL NOT NULL,
                    transaction_type TEXT NOT NULL,
                    merchant_name TEXT,
                    category TEXT DEFAULT 'Uncategorized',
                    balance REAL,
                    confidence_score REAL DEFAULT 1.0,
                    raw_text TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions_raw(date);
                CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions_raw(transaction_type);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_statements_hash ON statements(file_hash);

                CREATE TABLE IF NOT EXISTS transactions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_id              INTEGER NOT NULL UNIQUE,
                    statement_id        INTEGER,
                    bank_name           TEXT,
                    date                TEXT    NOT NULL,
                    description         TEXT    NOT NULL,
                    clean_description   TEXT,
                    amount              REAL    NOT NULL,
                    transaction_type    TEXT    NOT NULL
                            CHECK(transaction_type IN ('DEBIT','CREDIT','UNKNOWN')),
                    balance             REAL,
                    merchant_name       TEXT,
                    merchant_raw        TEXT,
                    category            TEXT    DEFAULT 'Uncategorized',
                    subcategory         TEXT,
                    location            TEXT,
                    confidence_score    REAL    DEFAULT 0.0,
                    enrichment_method   TEXT,
                    enriched_at         TEXT    DEFAULT (datetime('now')),
                    FOREIGN KEY (raw_id)       REFERENCES transactions_raw(id),
                    FOREIGN KEY (statement_id) REFERENCES statements(id)
                );
                CREATE INDEX IF NOT EXISTS idx_txn_date        ON transactions(date);
                CREATE INDEX IF NOT EXISTS idx_txn_type        ON transactions(transaction_type);
                CREATE INDEX IF NOT EXISTS idx_txn_merchant    ON transactions(merchant_name);
                CREATE INDEX IF NOT EXISTS idx_txn_category    ON transactions(category);
                CREATE INDEX IF NOT EXISTS idx_txn_subcategory ON transactions(subcategory);
                CREATE INDEX IF NOT EXISTS idx_txn_raw_id      ON transactions(raw_id);
            """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Apply any schema migrations needed for existing databases."""
        existing = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(transactions_raw)").fetchall()
        }
        if "bank_name" not in existing:
            self.conn.execute("ALTER TABLE transactions_raw ADD COLUMN bank_name TEXT")
            self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Supabase client (lazy, optional)                                    #
    # ------------------------------------------------------------------ #

    def _get_supa(self):
        """
        Lazy-load the Supabase client on first write attempt.
        Returns None if credentials are unavailable (offline / test mode).
        """
        if self._supa is not None:
            return self._supa
        try:
            import subprocess
            from supabase import create_client

            def _kc(service, account):
                r = subprocess.run(
                    ["security", "find-generic-password",
                     "-s", service, "-a", account, "-w"],
                    capture_output=True, text=True, check=True
                )
                return r.stdout.strip()

            url = _kc("supabase_bank", "url")
            key = _kc("supabase_bank", "anon_key")
            self._supa = create_client(url, key)
            log.debug("Supabase client initialised")
        except Exception as e:
            log.warning("Supabase unavailable — writes will be SQLite-only: %s", e)
            self._supa = False          # False = "tried and failed", skip retrying
        return self._supa or None

    # ------------------------------------------------------------------ #
    #  Duplicate detection                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_hash(filepath: str) -> str:
        """Return SHA-256 hex digest of a file's contents."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def check_duplicate(self, filepath: str) -> str:
        """
        Compute the file's SHA-256 and check the statements table.
        Raises DuplicateStatementError if the hash already exists.
        Returns the hash so the caller can reuse it.
        """
        file_hash = self.compute_hash(filepath)
        row = self.conn.execute(
            "SELECT original_path, import_date FROM statements WHERE file_hash = ?",
            (file_hash,)
        ).fetchone()

        if row:
            raise DuplicateStatementError(
                filepath=filepath,
                original_path=row["original_path"],
                import_date=row["import_date"],
            )
        return file_hash

    # ------------------------------------------------------------------ #
    #  Write operations — dual-write: SQLite first, then Supabase         #
    #  Rolls back SQLite savepoint if Supabase fails.                     #
    # ------------------------------------------------------------------ #

    def save_statement(self, filepath: str, bank_name: str = None,
                       statement_date: str = None,
                       file_hash: str = None) -> int:
        """
        Save a statement record and dual-write to Supabase.
        Caller should call check_duplicate() first.
        Returns the new statement id.
        """
        if file_hash is None:
            file_hash = self.compute_hash(filepath)

        filename     = os.path.basename(filepath)
        original_path = os.path.abspath(filepath)

        # ── Step 1: write to SQLite inside a savepoint ────────────────
        savepoint = "sp_save_statement"
        self.conn.execute(f"SAVEPOINT {savepoint}")
        try:
            cursor = self.conn.execute(
                """INSERT INTO statements
                   (filename, original_path, file_hash, bank_name, statement_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (filename, original_path, file_hash, bank_name, statement_date),
            )
            stmt_id = cursor.lastrowid

            # ── Step 2: write to Supabase ─────────────────────────────
            supa = self._get_supa()
            if supa:
                row = {
                    "id":            stmt_id,
                    "filename":      filename,
                    "original_path": original_path,
                    "file_hash":     file_hash,
                    "bank_name":     bank_name,
                    "statement_date": statement_date,
                }
                result = supa.table("statements").upsert(
                    row, on_conflict="file_hash"
                ).execute()
                if not result.data:
                    raise RuntimeError("Supabase upsert returned no data for statements")
                log.debug("Supabase: saved statement id=%s", stmt_id)

            # ── Both succeeded: commit SQLite ─────────────────────────
            self.conn.execute(f"RELEASE {savepoint}")
            self.conn.commit()

        except Exception as e:
            self.conn.execute(f"ROLLBACK TO {savepoint}")
            self.conn.execute(f"RELEASE {savepoint}")
            log.error("save_statement failed — SQLite rolled back: %s", e)
            raise

        return stmt_id

    def save_transactions(self, transactions: List[Dict],
                          statement_id: int = None,
                          bank_name: str = None) -> int:
        """
        Save a list of raw transactions and dual-write to Supabase.
        Returns count saved.
        All rows for a statement are written atomically — if Supabase
        fails, all SQLite inserts for this batch are rolled back.
        """
        if not transactions:
            return 0

        rows_to_insert = []
        for t in transactions:
            rows_to_insert.append((
                statement_id,
                t.get("bank_name", bank_name),
                t.get("date", ""),
                t.get("description", ""),
                float(t.get("amount", 0)),
                t.get("transaction_type", "UNKNOWN"),
                t.get("merchant_name", ""),
                t.get("category", "Uncategorized"),
                t.get("balance"),
                t.get("confidence_score", 1.0),
                t.get("raw_text", ""),
            ))

        # ── Step 1: write all rows to SQLite inside a savepoint ───────
        savepoint = "sp_save_transactions"
        self.conn.execute(f"SAVEPOINT {savepoint}")
        saved_ids = []
        try:
            for row in rows_to_insert:
                try:
                    cursor = self.conn.execute(
                        """INSERT INTO transactions_raw
                           (statement_id, bank_name, date, description, amount,
                            transaction_type, merchant_name, category, balance,
                            confidence_score, raw_text)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        row,
                    )
                    saved_ids.append(cursor.lastrowid)
                except Exception as e:
                    log.warning("Skipped one transaction (SQLite): %s", e)

            # ── Step 2: write to Supabase ─────────────────────────────
            supa = self._get_supa()
            if supa and saved_ids:
                # Re-fetch the rows we just inserted so we have their ids
                placeholders = ",".join("?" * len(saved_ids))
                inserted = self.conn.execute(
                    f"SELECT * FROM transactions_raw WHERE id IN ({placeholders})",
                    saved_ids,
                ).fetchall()

                supa_rows = [dict(r) for r in inserted]
                # Chunk to stay under Supabase 1 MB limit
                CHUNK = 500
                for i in range(0, len(supa_rows), CHUNK):
                    chunk = supa_rows[i:i + CHUNK]
                    result = supa.table("transactions_raw").upsert(
                        chunk, on_conflict="id"
                    ).execute()
                    if result.data is None:
                        raise RuntimeError(
                            f"Supabase upsert returned no data for transactions_raw chunk {i}"
                        )
                log.debug("Supabase: saved %d transactions_raw rows", len(supa_rows))

            # ── Both succeeded: commit SQLite ─────────────────────────
            self.conn.execute(f"RELEASE {savepoint}")
            self.conn.commit()

        except Exception as e:
            self.conn.execute(f"ROLLBACK TO {savepoint}")
            self.conn.execute(f"RELEASE {savepoint}")
            log.error("save_transactions failed — SQLite rolled back: %s", e)
            raise

        saved = len(saved_ids)
        if statement_id and saved:
            self._update_statement_totals(statement_id)
        return saved

    def _update_statement_totals(self, statement_id: int):
        """
        Update denormalised totals on the statements row.
        Dual-writes the updated statement row to Supabase.
        """
        self.conn.execute("""
            UPDATE statements SET
                total_transactions = (SELECT COUNT(*) FROM transactions_raw
                                      WHERE statement_id = ?),
                total_debits  = (SELECT COALESCE(SUM(amount), 0)
                                 FROM transactions_raw
                                 WHERE statement_id = ? AND transaction_type = 'DEBIT'),
                total_credits = (SELECT COALESCE(SUM(amount), 0)
                                 FROM transactions_raw
                                 WHERE statement_id = ? AND transaction_type = 'CREDIT')
            WHERE id = ?
        """, (statement_id, statement_id, statement_id, statement_id))
        self.conn.commit()

        # Sync updated totals to Supabase
        supa = self._get_supa()
        if supa:
            try:
                row = self.conn.execute(
                    "SELECT id, total_transactions, total_debits, total_credits "
                    "FROM statements WHERE id = ?",
                    (statement_id,)
                ).fetchone()
                if row:
                    supa.table("statements").update({
                        "total_transactions": row["total_transactions"],
                        "total_debits":       row["total_debits"],
                        "total_credits":      row["total_credits"],
                    }).eq("id", statement_id).execute()
                    log.debug("Supabase: updated totals for statement id=%s", statement_id)
            except Exception as e:
                # Totals sync failure is non-fatal — data is still consistent
                log.warning("Supabase totals sync failed for statement %s: %s",
                            statement_id, e)

    # ------------------------------------------------------------------ #
    #  Read operations — SQLite only (legacy / internal reads)            #
    # ------------------------------------------------------------------ #

    def _active_table(self) -> str:
        """Return 'transactions' if it has rows, else fall back to 'transactions_raw'."""
        try:
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
            ).fetchone()
            if row:
                count = self.conn.execute(
                    "SELECT COUNT(*) FROM transactions"
                ).fetchone()[0]
                if count > 0:
                    return "transactions"
        except Exception:
            pass
        return "transactions_raw"

    def get_transactions(self, limit: int = 100,
                         transaction_type: str = None,
                         start_date: str = None,
                         end_date: str = None,
                         merchant: str = None,
                         bank_name: str = None) -> List[Dict]:
        table = self._active_table()
        query = f"SELECT * FROM {table} WHERE 1=1"
        params: List[Any] = []

        if transaction_type:
            query += " AND transaction_type = ?"
            params.append(transaction_type.upper())
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if merchant:
            query += " AND merchant_name LIKE ?"
            params.append(f"%{merchant}%")
        if bank_name:
            query += " AND bank_name = ?"
            params.append(bank_name.lower())

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def get_statements(self) -> List[Dict]:
        """List all imported statements."""
        return [dict(r) for r in self.conn.execute(
            "SELECT id, filename, bank_name, import_date, "
            "total_transactions, total_debits, total_credits, file_hash "
            "FROM statements ORDER BY import_date DESC"
        ).fetchall()]

    def get_summary_stats(self) -> Dict[str, Any]:
        row = dict(self.conn.execute("""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN transaction_type='DEBIT'  THEN amount ELSE 0 END), 0) as total_debits,
                COALESCE(SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END), 0) as total_credits,
                COALESCE(AVG(CASE WHEN transaction_type='DEBIT'  THEN amount END), 0)        as avg_debit,
                COALESCE(MAX(amount), 0) as largest_transaction
            FROM transactions_raw
        """).fetchone())
        row["net"] = row["total_credits"] - row["total_debits"]
        return row

    def get_largest_transaction(self) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM transactions_raw ORDER BY amount DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_transactions_by_merchant(self, merchant: str) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM transactions_raw WHERE merchant_name LIKE ? ORDER BY date DESC",
            (f"%{merchant}%",)
        ).fetchall()]

    def get_monthly_spending(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("""
            SELECT
                substr(date, 1, 7) as month,
                SUM(CASE WHEN transaction_type='DEBIT'  THEN amount ELSE 0 END) as spending,
                SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END) as income,
                COUNT(*) as transaction_count
            FROM transactions_raw
            GROUP BY month ORDER BY month DESC
        """).fetchall()]

    def clear_all(self):
        self.conn.execute("DELETE FROM transactions_raw")
        self.conn.execute("DELETE FROM statements")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self):
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<DatabaseManager path={self.db_path!r}>"
