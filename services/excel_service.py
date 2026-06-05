"""
services/excel_service.py
--------------------------
Excel file ingestion, row storage, and lookup for the Local Employee Knowledge Assistant.

Excel is fundamentally different from prose documents — the unit of meaning
is a row, not a paragraph. This service handles Excel separately:

  Ingestion:
    Each row stored as JSONB in excel_rows table.
    Each row also stringified for BM25 keyword search.

  Lookup (Type 1 — row search):
    User asks: "Is order 1042 in the sheet?" / "Find Ravi Kumar's invoice"
    → BM25 search on stringified rows + optional SQL filter for exact IDs
    → Returns matching rows as a table

  Aggregation (Type 2 — computed answers):
    User asks: "How many pending orders?" / "Total sales for March"
    → LLM generates SQL → executed against excel_rows table
    → Returns computed result + LLM natural language summary

  Metadata (Type 3 — structure questions):
    Handled by metadata_service.py — sheet names, row count, column names
    stored at ingestion time.

Schema:
  excel_rows
    id, document_id, sheet_name, row_number, row_data (JSONB), row_text (TEXT)
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def ensure_excel_table() -> None:
    """Create excel_rows table if it does not exist. Idempotent."""
    from services.database_service import _get_raw_connection

    with _get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS excel_rows (
                    id          SERIAL PRIMARY KEY,
                    document_id INTEGER  NOT NULL
                                    REFERENCES documents(id) ON DELETE CASCADE,
                    sheet_name  TEXT     NOT NULL,
                    row_number  INTEGER  NOT NULL,
                    row_data    JSONB    NOT NULL,
                    row_text    TEXT     NOT NULL,
                    CONSTRAINT excel_rows_unique UNIQUE (document_id, sheet_name, row_number)
                );
            """)
            # Full-text search index on row_text for fast keyword search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS excel_rows_text_idx
                ON excel_rows USING gin(to_tsvector('english', row_text));
            """)

    logger.debug("excel_rows table and index ensured.")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_excel(filepath: str, document_id: int) -> dict:
    """
    Load an Excel file and store all rows in the excel_rows table.

    Args:
        filepath    : Path to the .xlsx or .xls file.
        document_id : The document's DB row ID.

    Returns:
        dict with sheet_count, total_rows, sheets (list of sheet summaries)
    """
    import openpyxl
    import psycopg2.extras
    from services.database_service import _get_connection

    wb    = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    total = 0
    sheets_summary = []

    with _get_connection() as conn:
        with conn.cursor() as cur:
            for sheet_name in wb.sheetnames:
                ws   = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))

                if not rows:
                    continue

                # First row = headers
                headers  = [str(c).strip() if c is not None else f"col_{i}"
                            for i, c in enumerate(rows[0])]
                data_rows = rows[1:]
                sheet_row_count = 0

                for row_idx, row in enumerate(data_rows, start=1):
                    # Skip completely empty rows
                    if all(v is None for v in row):
                        continue

                    # Build row dict — header: value
                    row_dict = {}
                    for h, v in zip(headers, row):
                        if v is not None:
                            row_dict[h] = v if not hasattr(v, 'isoformat') else str(v)

                    if not row_dict:
                        continue

                    # Stringify for BM25 / full-text search
                    row_text = _stringify_row(row_dict)

                    cur.execute("""
                        INSERT INTO excel_rows
                            (document_id, sheet_name, row_number, row_data, row_text)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (document_id, sheet_name, row_number) DO UPDATE
                            SET row_data = EXCLUDED.row_data,
                                row_text = EXCLUDED.row_text;
                    """, (
                        document_id,
                        sheet_name,
                        row_idx,
                        json.dumps(row_dict),
                        row_text,
                    ))
                    sheet_row_count += 1
                    total += 1

                sheets_summary.append({
                    "sheet_name": sheet_name,
                    "headers":    headers,
                    "row_count":  sheet_row_count,
                })

    wb.close()

    logger.info(
        "Excel ingested: %d sheets, %d rows stored for document_id=%d.",
        len(sheets_summary), total, document_id,
    )
    return {
        "sheet_count": len(sheets_summary),
        "total_rows":  total,
        "sheets":      sheets_summary,
    }


def _stringify_row(row_dict: dict) -> str:
    """
    Convert a row dict to a searchable string.
    Format: "Column1: value1  Column2: value2  ..."
    """
    return "  ".join(f"{k}: {v}" for k, v in row_dict.items())


# ---------------------------------------------------------------------------
# Row lookup (Type 1)
# ---------------------------------------------------------------------------

def search_rows(
    query: str,
    document_ids: list = None,
    top_k: int = 20,
) -> list:
    """
    Search Excel rows by keyword using PostgreSQL full-text search.

    Args:
        query        : The user's natural language question.
        document_ids : Optional list of document IDs to restrict search.
        top_k        : Maximum rows to return.

    Returns:
        List of dicts: {document_id, sheet_name, row_number, row_data, row_text, rank}
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    # Extract search terms — strip common words
    search_terms = _extract_search_terms(query)
    if not search_terms:
        return []

    ts_query = " & ".join(search_terms)

    if document_ids:
        sql = """
            SELECT
                e.id, e.document_id, e.sheet_name, e.row_number,
                e.row_data, e.row_text,
                d.filename,
                ts_rank(to_tsvector('english', e.row_text),
                        to_tsquery('english', %s)) AS rank
            FROM excel_rows e
            JOIN documents d ON d.id = e.document_id
            WHERE e.document_id = ANY(%s)
              AND to_tsvector('english', e.row_text) @@ to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        params = (ts_query, document_ids, ts_query, top_k)
    else:
        sql = """
            SELECT
                e.id, e.document_id, e.sheet_name, e.row_number,
                e.row_data, e.row_text,
                d.filename,
                ts_rank(to_tsvector('english', e.row_text),
                        to_tsquery('english', %s)) AS rank
            FROM excel_rows e
            JOIN documents d ON d.id = e.document_id
            WHERE to_tsvector('english', e.row_text) @@ to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        params = (ts_query, ts_query, top_k)

    try:
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []
        for r in rows:
            row = dict(r)
            if isinstance(row["row_data"], str):
                row["row_data"] = json.loads(row["row_data"])
            results.append(row)

        logger.info(
            "Excel row search: '%s' → %d rows found.", query[:60], len(results)
        )
        return results

    except Exception as exc:
        logger.warning("Excel full-text search failed: %s. Trying fallback.", exc)
        return _fallback_search(query, document_ids, top_k)


def _fallback_search(query: str, document_ids: list, top_k: int) -> list:
    """
    Simple ILIKE fallback when full-text search fails
    (e.g. special characters in the query).
    """
    import psycopg2.extras
    from services.database_service import _get_connection

    terms = _extract_search_terms(query)
    if not terms:
        return []

    pattern = f"%{terms[0]}%"
    scope   = "AND e.document_id = ANY(%s)" if document_ids else ""
    params  = [pattern, document_ids, top_k] if document_ids else [pattern, top_k]

    sql = f"""
        SELECT e.id, e.document_id, e.sheet_name, e.row_number,
               e.row_data, e.row_text, d.filename, 1.0 AS rank
        FROM excel_rows e
        JOIN documents d ON d.id = e.document_id
        WHERE e.row_text ILIKE %s {scope}
        LIMIT %s;
    """

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        if isinstance(r["row_data"], str):
            r["row_data"] = json.loads(r["row_data"])
    return rows


def _extract_search_terms(query: str) -> list:
    """
    Extract meaningful search terms from a query.
    Removes common filler words and returns tokens suitable for ts_query.
    """
    import re
    stop = {"is", "are", "the", "a", "an", "in", "of", "for", "to", "there",
            "any", "find", "show", "get", "what", "does", "do", "has", "have",
            "me", "this", "that", "which", "where", "how", "can", "could"}
    tokens = re.findall(r"[a-zA-Z0-9\-]+", query)
    return [t for t in tokens if t.lower() not in stop and len(t) >= 2]


# ---------------------------------------------------------------------------
# Aggregation with LLM-generated SQL (Type 2)
# ---------------------------------------------------------------------------

def answer_aggregation(query: str, document_id: int = None) -> dict:
    """
    Answer an aggregation question about Excel data using LLM-generated SQL.

    Flow:
        1. Fetch schema (sheet names, columns) for context
        2. LLM generates a SQL query for the excel_rows table
        3. Execute the SQL
        4. LLM generates a natural language summary

    Returns:
        dict with: answered, sql, result_rows, response
    """
    import psycopg2.extras
    from services.database_service import _get_connection, get_all_documents
    from services.llm_service import _call_ollama

    # Get schema context
    schema_parts = []
    docs = get_all_documents()
    for doc in docs:
        if document_id and doc["id"] != document_id:
            continue
        meta = {}
        try:
            from services.metadata_service import get_metadata
            meta = get_metadata(doc["id"])
        except Exception:
            pass
        if meta:
            schema_parts.append(f"File: {doc['filename']}")
            for k, v in meta.items():
                if "column" in k or "sheet" in k:
                    schema_parts.append(f"  {k}: {v}")

    schema_context = "\n".join(schema_parts) if schema_parts else "No schema available."

    # Prompt LLM to generate SQL
    sql_prompt = f"""You are a SQL expert. Generate a PostgreSQL SQL query to answer this question.

The data is stored in the excel_rows table with these columns:
  - document_id (INTEGER)
  - sheet_name  (TEXT)
  - row_number  (INTEGER)
  - row_data    (JSONB) — contains the actual column values as key-value pairs

Document schema context:
{schema_context}

To access a JSONB field, use: row_data->>'ColumnName'
To count rows: SELECT COUNT(*) FROM excel_rows WHERE ...
To sum: SELECT SUM((row_data->>'Amount')::numeric) FROM excel_rows WHERE ...

Question: {query}

Return ONLY the SQL query, no explanation, no markdown."""

    try:
        raw_sql = _call_ollama(sql_prompt).strip()
        # Clean up SQL (remove markdown code blocks if present)
        import re
        raw_sql = re.sub(r"```(?:sql)?", "", raw_sql).strip()

        # Safety: only allow SELECT queries
        if not raw_sql.upper().startswith("SELECT"):
            return {
                "answered": False,
                "response": "Could not generate a safe SQL query for this question.",
                "sql":      raw_sql,
                "result_rows": []
            }

        # Execute the SQL
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(raw_sql)
                result_rows = [dict(r) for r in cur.fetchall()]

        # LLM generates natural language summary
        summary_prompt = f"""Given this question: "{query}"
And these SQL results: {json.dumps(result_rows[:10])}
Write a clear, concise one-sentence answer in plain English."""

        response = _call_ollama(summary_prompt).strip()

        return {
            "answered":    True,
            "sql":         raw_sql,
            "result_rows": result_rows,
            "response":    response,
        }

    except Exception as exc:
        logger.warning("Aggregation query failed: %s", exc)
        return {
            "answered": False,
            "response": f"Could not compute the answer: {exc}",
            "sql":      "",
            "result_rows": []
        }


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

LOOKUP_KEYWORDS = {
    "find", "search", "show", "get", "is", "are", "exists", "exist",
    "look", "check", "there", "any", "which", "locate", "list",
    "order", "invoice", "bill", "record", "entry", "row",
}

AGGREGATION_KEYWORDS = {
    "how many", "count", "total", "sum", "average", "max", "min",
    "highest", "lowest", "most", "least", "percentage", "percent",
    "calculate", "compute", "aggregate",
}


def detect_excel_intent(query: str) -> str:
    """
    Classify an Excel-related query into intent type.

    Returns:
        "lookup"      — row search
        "aggregation" — computed answer
        "metadata"    — structure question (handled by metadata_service)
        "unknown"     — cannot determine
    """
    q = query.lower()

    from services.metadata_service import is_metadata_question
    if is_metadata_question(query):
        return "metadata"

    if any(phrase in q for phrase in AGGREGATION_KEYWORDS):
        return "aggregation"

    if any(word in q.split() for word in LOOKUP_KEYWORDS):
        return "lookup"

    return "lookup"   # default to lookup for Excel queries
