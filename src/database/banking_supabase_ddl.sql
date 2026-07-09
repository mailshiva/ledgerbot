
-- =========================================================================
-- Bank Statements (shared metadata)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_statements (
    id              BIGSERIAL PRIMARY KEY,
    filename        TEXT NOT NULL,
    original_path   TEXT NOT NULL,
    file_hash       TEXT NOT NULL UNIQUE,
    bank_name       TEXT NOT NULL,
    statement_period_start DATE,
    statement_period_end   DATE,
    import_date     TIMESTAMPTZ DEFAULT now(),
    total_transactions INTEGER DEFAULT 0,
    total_debits    NUMERIC(12,2) DEFAULT 0.0,
    total_credits   NUMERIC(12,2) DEFAULT 0.0
);

-- =========================================================================
-- Bank Transactions — Raw (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions_raw (
    id              BIGSERIAL PRIMARY KEY,
    statement_id    BIGINT NOT NULL REFERENCES bank_statements(id),
    bank_name       TEXT NOT NULL,
    account_type    TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date            DATE NOT NULL,
    description     TEXT NOT NULL,
    amount          NUMERIC(12,2) NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance         NUMERIC(12,2),
    merchant_name   TEXT,
    category        TEXT DEFAULT 'Uncategorized',
    confidence_score NUMERIC(3,2) DEFAULT 1.0,
    raw_text        TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Bank Transactions — Enriched (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions (
    id                  BIGSERIAL PRIMARY KEY,
    raw_id              BIGINT NOT NULL UNIQUE REFERENCES bank_transactions_raw(id),
    statement_id        BIGINT REFERENCES bank_statements(id),
    bank_name           TEXT NOT NULL,
    account_type        TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date                DATE NOT NULL,
    description         TEXT NOT NULL,
    clean_description   TEXT,
    amount              NUMERIC(12,2) NOT NULL,
    transaction_type    TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance             NUMERIC(12,2),
    merchant_name       TEXT,
    merchant_raw        TEXT,
    category            TEXT DEFAULT 'Uncategorized',
    subcategory         TEXT,
    location            TEXT,
    confidence_score    NUMERIC(3,2) DEFAULT 0.0,
    enrichment_method   TEXT,
    enriched_at         TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Loan Transactions — Raw
-- =========================================================================
CREATE TABLE IF NOT EXISTS loan_transactions_raw (
    id              BIGSERIAL PRIMARY KEY,
    statement_id    BIGINT NOT NULL REFERENCES bank_statements(id),
    bank_name       TEXT NOT NULL,
    loan_identifier TEXT,
    date            DATE NOT NULL,
    description     TEXT NOT NULL,
    payment_amount  NUMERIC(12,2),
    principal_amount NUMERIC(12,2),
    interest_amount NUMERIC(12,2),
    balance         NUMERIC(12,2),
    raw_text        TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Loan Transactions — Enriched
-- =========================================================================
CREATE TABLE IF NOT EXISTS loan_transactions (
    id                  BIGSERIAL PRIMARY KEY,
    raw_id              BIGINT NOT NULL UNIQUE REFERENCES loan_transactions_raw(id),
    statement_id        BIGINT REFERENCES bank_statements(id),
    bank_name           TEXT NOT NULL,
    loan_identifier     TEXT,
    date                DATE NOT NULL,
    description         TEXT NOT NULL,
    clean_description   TEXT,
    payment_amount      NUMERIC(12,2),
    principal_amount    NUMERIC(12,2),
    interest_amount     NUMERIC(12,2),
    balance             NUMERIC(12,2),
    category            TEXT DEFAULT 'Loan Payment',
    subcategory         TEXT,
    confidence_score    NUMERIC(3,2) DEFAULT 0.0,
    enrichment_method   TEXT,
    enriched_at         TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Indexes
-- =========================================================================
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_date ON bank_transactions_raw(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_account_type ON bank_transactions_raw(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_bank_name ON bank_transactions_raw(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_stmt ON bank_transactions_raw(statement_id);

CREATE INDEX IF NOT EXISTS idx_bank_txn_date ON bank_transactions(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_account_type ON bank_transactions(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_bank_name ON bank_transactions(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_category ON bank_transactions(category);
CREATE INDEX IF NOT EXISTS idx_bank_txn_merchant ON bank_transactions(merchant_name);

CREATE INDEX IF NOT EXISTS idx_loan_txn_raw_date ON loan_transactions_raw(date);
CREATE INDEX IF NOT EXISTS idx_loan_txn_raw_stmt ON loan_transactions_raw(statement_id);
CREATE INDEX IF NOT EXISTS idx_loan_txn_date ON loan_transactions(date);

-- =========================================================================
-- Enable Row Level Security (recommended for Supabase)
-- =========================================================================
ALTER TABLE bank_statements ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank_transactions_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_transactions_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_transactions ENABLE ROW LEVEL SECURITY;
