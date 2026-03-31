# Bank Statement Parser — Week 1

A fully functional bank statement parser that extracts transactions from
PDF and CSV files, stores them in a local SQLite database, watches a folder
for new statements, and ships with a full test suite.

## Supported Banks

| Bank | Format | Notes |
|---|---|---|
| Robinhood | PDF | MM/DD dates, trailing `-` for credits |
| Citi | PDF | Two-column layout, `$amount` format |
| Bank of America | PDF | 6-column with ref numbers |
| Capital One | PDF | Month-name dates (`Aug 6`) |
| Any bank | CSV | Auto-detects column layout |

---

## Quick Start

```bash
# 1. Set up virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run self-test
python transaction_processor_manual.py test

# 4. Check config (creates ~/.bank_parser/config.yaml on first run)
python transaction_processor_manual.py config
```

---

## CLI Commands

```bash
python transaction_processor_manual.py config              # show config + DB path
python transaction_processor_manual.py test                # self-test with sample CSV
python transaction_processor_manual.py process <file>      # parse and import a PDF or CSV
python transaction_processor_manual.py stats               # DB summary statistics
python transaction_processor_manual.py recent [n]          # show last n transactions (default 10)
python transaction_processor_manual.py merchant <name>     # filter by merchant
python transaction_processor_manual.py monthly             # monthly spending breakdown
python transaction_processor_manual.py statements          # list all imported statement files
python transaction_processor_manual.py clear               # delete all data (with confirmation)
```

---

## Statement Watcher

The watcher scans a folder for new PDF statements, skips files already
imported (detected by SHA-256 hash), and imports new ones.

```bash
# Run manually
python watcher.py

# Override the folder for this run only
python watcher.py /path/to/statements

# Preview without writing to DB
python watcher.py --dry-run

# Schedule with cron — every 30 minutes
*/30 * * * * /path/to/venv/bin/python /path/to/credit_card_transactions/watcher.py
```

Configure the watch folder in `~/.bank_parser/config.yaml`:

```yaml
watcher:
  folder:   /Users/you/Downloads/statements
  log_file: /Users/you/.bank_parser/watcher.log
```

---

## Configuration

Config lives at `~/.bank_parser/config.yaml` (created automatically on first run).

```yaml
database:
  path: /Users/you/.bank_parser/transactions.db   # outside the project folder

watcher:
  folder:   /Users/you/Downloads/statements
  log_file: /Users/you/.bank_parser/watcher.log

parser:
  default_year: 2026          # fallback for PDFs with no year in dates
  confidence_threshold: 0.5

cli:
  date_format: "%Y-%m-%d"
```

---

## Running Tests

Tests use Python's built-in `unittest` — no extra dependencies needed.
When `pytest` is installed they also run under `pytest`.

```bash
# All tests
python -m unittest tests.test_unit tests.test_integration -v

# Unit tests only (fast, no PDF parsing)
python -m unittest tests.test_unit -v

# Integration tests only (parses all sample PDFs)
python -m unittest tests.test_integration -v

# With pytest (if installed)
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test coverage

| Suite | Tests | What's covered |
|---|---|---|
| Unit | 35 | Date/amount/type extraction, merchant standardisation (11 merchants), categorisation (6 categories), bank detection (10 patterns), year detection, DB schema, hash stability, duplicate detection, bank filtering, config defaults |
| Integration | 26 | Parse accuracy for all 5 banks (exact $ totals), field/date/amount validation, DB round-trip, bank isolation, renamed-file duplicate detection, watcher scan/skip/dry-run/incremental, CSV round-trip |

---

## Project Structure

```
bank_statement_parser/
├── week1_cli.py               ← main CLI entry point
├── watcher.py                 ← statement folder watcher
├── requirements.txt
├── README.md
│
├── data/
│   └── samples/               ← sample statements for tests
│
├── src/
│   ├── config.py              ← YAML config loader
│   ├── watcher.py             ← core watcher logic
│   ├── parsers/
│   │   ├── csv_parser.py      ← CSV parsing
│   │   └── pdf_parser.py      ← PDF extraction (4 banks)
│   ├── extractors/
│   │   └── rule_based.py      ← regex transaction extraction
│   └── database/
│       ├── schema.sql
│       └── db_manager.py      ← SQLite operations
│
└── tests/
    ├── conftest.py            ← shared fixtures and helpers
    ├── test_unit.py           ← 35 unit tests
    └── test_integration.py    ← 26 integration tests
```
