"""
Database Manager - Week 1
Handles all SQLite operations. DB path is read from ~/.bank_parser/config.yaml.
Duplicate file detection via SHA-256 hash before any write.
"""

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


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
                db_path = str(Path.home() / ".bank_parser" / "transactions.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._initialize_schema()

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
                    statement_date TEXT,
                    import_date TEXT DEFAULT (datetime('now')),
                    total_transactions INTEGER DEFAULT 0,
                    total_debits REAL DEFAULT 0.0,
                    total_credits REAL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS transactions (
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
                CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
                CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_statements_hash ON statements(file_hash);
            """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Apply any schema migrations needed for existing databases."""
        existing = {
            row[1] for row in
            self.conn.execute("PRAGMA table_info(transactions)").fetchall()
        }
        if "bank_name" not in existing:
            self.conn.execute("ALTER TABLE transactions ADD COLUMN bank_name TEXT")
            self.conn.commit()

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

    def check_duplicate(self, filepath: str) -> None:
        """
        Compute the file's SHA-256 and check the statements table.
        Raises DuplicateStatementError if the hash already exists.
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
        return file_hash   # return hash so caller can reuse it

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def save_statement(self, filepath: str, bank_name: str = None,
                       statement_date: str = None,
                       file_hash: str = None) -> int:
        """
        Save a statement record. Caller should call check_duplicate() first.
        file_hash may be passed in to avoid re-computing it.
        """
        if file_hash is None:
            file_hash = self.compute_hash(filepath)

        cursor = self.conn.execute(
            """INSERT INTO statements
               (filename, original_path, file_hash, bank_name, statement_date)
               VALUES (?, ?, ?, ?, ?)""",
            (
                os.path.basename(filepath),
                os.path.abspath(filepath),
                file_hash,
                bank_name,
                statement_date,
            )
        )
        self.conn.commit()
        return cursor.lastrowid

    def save_transactions(self, transactions: List[Dict],
                          statement_id: int = None,
                          bank_name: str = None) -> int:
        """Save a list of transactions. Returns count saved."""
        saved = 0
        for t in transactions:
            try:
                self.conn.execute(
                    """INSERT INTO transactions
                       (statement_id, bank_name, date, description, amount,
                        transaction_type, merchant_name, category, balance,
                        confidence_score, raw_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        statement_id,
                        t.get("bank_name", bank_name),   # per-row override or caller-supplied
                        t.get("date", ""),
                        t.get("description", ""),
                        float(t.get("amount", 0)),
                        t.get("transaction_type", "UNKNOWN"),
                        t.get("merchant_name", ""),
                        t.get("category", "Uncategorized"),
                        t.get("balance"),
                        t.get("confidence_score", 1.0),
                        t.get("raw_text", ""),
                    )
                )
                saved += 1
            except Exception as e:
                print(f"  ⚠️  Skipped transaction: {e}")

        self.conn.commit()
        if statement_id:
            self._update_statement_totals(statement_id)
        return saved

    def _update_statement_totals(self, statement_id: int):
        self.conn.execute("""
            UPDATE statements SET
                total_transactions = (SELECT COUNT(*) FROM transactions WHERE statement_id = ?),
                total_debits  = (SELECT COALESCE(SUM(amount), 0) FROM transactions
                                 WHERE statement_id = ? AND transaction_type = 'DEBIT'),
                total_credits = (SELECT COALESCE(SUM(amount), 0) FROM transactions
                                 WHERE statement_id = ? AND transaction_type = 'CREDIT')
            WHERE id = ?
        """, (statement_id, statement_id, statement_id, statement_id))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Read operations                                                     #
    # ------------------------------------------------------------------ #

    def get_transactions(self, limit: int = 100,
                         transaction_type: str = None,
                         start_date: str = None,
                         end_date: str = None,
                         merchant: str = None,
                         bank_name: str = None) -> List[Dict]:
        query = "SELECT * FROM transactions WHERE 1=1"
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
            FROM transactions
        """).fetchone())
        row["net"] = row["total_credits"] - row["total_debits"]
        return row

    def get_largest_transaction(self) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM transactions ORDER BY amount DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_transactions_by_merchant(self, merchant: str) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM transactions WHERE merchant_name LIKE ? ORDER BY date DESC",
            (f"%{merchant}%",)
        ).fetchall()]

    def get_monthly_spending(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("""
            SELECT
                substr(date, 1, 7) as month,
                SUM(CASE WHEN transaction_type='DEBIT'  THEN amount ELSE 0 END) as spending,
                SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END) as income,
                COUNT(*) as transaction_count
            FROM transactions
            GROUP BY month ORDER BY month DESC
        """).fetchall()]

    def clear_all(self):
        self.conn.execute("DELETE FROM transactions")
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
