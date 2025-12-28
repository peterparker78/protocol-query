# Protocol Query CLI

Interactive clinical protocol querying CLI with hybrid search and LLM-powered analysis.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Initialize the database
protocol-query config init

# Ingest a protocol
protocol-query ingest add ./protocol.pdf --protocol-id STUDY-001

# Search
protocol-query query search "inclusion criteria"

# What-if analysis (requires ANTHROPIC_API_KEY)
protocol-query query what-if "patient has diabetes" --protocol STUDY-001

# Compare protocols
protocol-query compare eligibility STUDY-001 STUDY-002
```

## Configuration

Set environment variables or create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
PROTOCOL_DB_PATH=./data/protocols.db
```
