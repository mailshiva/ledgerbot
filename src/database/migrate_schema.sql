-- Week 2 enrichment migration
-- Safe to run on an existing DB that already has transactions_raw.
-- Run once via: python -m src.db.migrate  OR  sqlite3 your.db < migrate_schema.sql

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

-- 3. Back-fill: add subcategory column to transactions_raw if it doesn't exist yet.
--    Python's migrate() wraps this in a try/except so it's always safe to re-run.
-- ALTER TABLE transactions_raw ADD COLUMN subcategory TEXT;
