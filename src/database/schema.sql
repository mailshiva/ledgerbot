-- Bank Statement Parser - Database Schema
-- Week 1: Foundation

CREATE TABLE IF NOT EXISTS statements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    filename            TEXT    NOT NULL,
    original_path       TEXT    NOT NULL,           -- full path at import time
    file_hash           TEXT    NOT NULL UNIQUE,    -- SHA-256 of file contents
    bank_name           TEXT,
    account_number      TEXT,
    statement_date      TEXT,
    import_date         TEXT    DEFAULT (datetime('now')),
    total_transactions  INTEGER DEFAULT 0,
    total_debits        REAL    DEFAULT 0.0,
    total_credits       REAL    DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id        INTEGER,
    bank_name           TEXT,                          -- denormalised for easy filtering
    date                TEXT    NOT NULL,
    description         TEXT    NOT NULL,
    amount              REAL    NOT NULL,
    transaction_type    TEXT    NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    merchant_name       TEXT,
    category            TEXT    DEFAULT 'Uncategorized',
    balance             REAL,
    confidence_score    REAL    DEFAULT 1.0,
    raw_text            TEXT,
    created_at          TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (statement_id) REFERENCES statements(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_date       ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_type       ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant   ON transactions(merchant_name);
CREATE INDEX IF NOT EXISTS idx_transactions_category   ON transactions(category);
CREATE UNIQUE INDEX IF NOT EXISTS idx_statements_hash  ON statements(file_hash);

-- Migration: add bank_name to existing databases
-- Safe to run multiple times (ALTER TABLE ignores if column exists via try/catch in Python)
