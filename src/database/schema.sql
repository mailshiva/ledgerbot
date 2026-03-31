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

CREATE TABLE IF NOT EXISTS transactions_raw (
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
CREATE INDEX IF NOT EXISTS idx_transactions_date       ON transactions_raw(date);
CREATE INDEX IF NOT EXISTS idx_transactions_type       ON transactions_raw(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant   ON transactions_raw(merchant_name);
CREATE INDEX IF NOT EXISTS idx_transactions_category   ON transactions_raw(category);
CREATE UNIQUE INDEX IF NOT EXISTS idx_statements_hash  ON statements(file_hash);

-- Migration: add bank_name to existing databases
-- Safe to run multiple times (ALTER TABLE ignores if column exists via try/catch in Python)

-- 1. Enriched transactions table (output of the NLP pipeline)
CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id              INTEGER NOT NULL UNIQUE,        -- FK back to transactions_raw
    statement_id        INTEGER,
    bank_name           TEXT,
    date                TEXT    NOT NULL,
    description         TEXT    NOT NULL,               -- original, un-cleaned
    clean_description   TEXT,                           -- after regex cleaning
    amount              REAL    NOT NULL,
    transaction_type    TEXT    NOT NULL
                            CHECK(transaction_type IN ('DEBIT','CREDIT','UNKNOWN')),
    balance             REAL,

    -- NLP-enriched columns
    merchant_name       TEXT,
    merchant_raw        TEXT,                           -- pre-normalisation merchant token
    category            TEXT    DEFAULT 'Uncategorized',
    subcategory         TEXT,                           -- e.g. 'Streaming' under 'Entertainment'
    location            TEXT,                           -- city / state extracted by spaCy GPE
    confidence_score    REAL    DEFAULT 0.0,            -- composite 0-1

    -- provenance
    enrichment_method   TEXT,   -- 'substring' | 'fuzzy' | 'spacy' | 'rules' | 'fallback'
    enriched_at         TEXT    DEFAULT (datetime('now')),

    FOREIGN KEY (raw_id)      REFERENCES transactions_raw(id),
    FOREIGN KEY (statement_id) REFERENCES statements(id)
);

-- 2. Indexes for the enriched table
CREATE INDEX IF NOT EXISTS idx_txn_date       ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_txn_type       ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_txn_merchant   ON transactions(merchant_name);
CREATE INDEX IF NOT EXISTS idx_txn_category   ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_txn_subcategory ON transactions(subcategory);
CREATE INDEX IF NOT EXISTS idx_txn_raw_id     ON transactions(raw_id);
