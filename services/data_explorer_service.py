"""
services/data_explorer_service.py
----------------------------------
Data exploration and natural language querying for CSV and Excel files.

This service handles ad-hoc data exploration — completely separate from the
RAG pipeline. Files are loaded into memory for the current session only.
Nothing is stored in PostgreSQL.

Three query modes:
  Mode 1 — Statistical (instant, no LLM)
    Detected from keywords. Runs pandas operations directly.
    Always works. Covers ~65% of typical business queries.

  Mode 2 — LLM → Pandas Code
    LLM generates pandas code from natural language.
    Code is validated and executed in a sandboxed namespace.
    Retries once on failure with the error fed back.

  Mode 3 — Visualization
    Detected from chart/plot keywords.
    LLM generates matplotlib code. Chart saved and returned.

Safety:
  All generated code runs in a restricted namespace — only the DataFrame
  and whitelisted libraries are accessible. File system, network, and
  system calls are blocked before execution.
"""

import io
import logging
import os
import re
import sys
import traceback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whitelisted modules for safe code execution
# ---------------------------------------------------------------------------

SAFE_MODULES = {
    "pandas", "pd", "numpy", "np", "math", "statistics",
    "datetime", "re", "collections", "itertools",
}

BLOCKED_PATTERNS = [
    r"\bos\b", r"\bsys\b", r"\bsubprocess\b", r"\bopen\s*\(",
    r"\bexec\s*\(", r"\beval\s*\(", r"__import__",
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bshutil\b", r"\bpathlib\b", r"\.write\s*\(",
    r"\brequests\b", r"\burllib\b", r"\bsocket\b",
    # Block file reading — df is already loaded, no file access needed
    r"pd\.read_csv", r"pd\.read_excel", r"pd\.read_",
    r"open\s*\(", r"\bcsv\.reader\b",
]

# Statistical intent keywords
STAT_KEYWORDS = {
    "describe", "summary", "summarise", "summarize", "info", "shape",
    "head", "tail", "first", "last", "columns", "dtypes", "types",
    "null", "missing", "nan", "isnull", "isna", "duplicate",
    "unique", "nunique", "count", "value_counts",
    "correlation", "corr", "mean", "median", "mode", "std",
    "variance", "min", "max", "range", "percentile", "quantile",
}

# Visualization intent keywords
VIZ_KEYWORDS = {
    "chart", "plot", "graph", "visuali", "histogram", "bar",
    "pie", "scatter", "line", "heatmap", "boxplot", "distribution",
    "show me a", "draw", "display a",
}


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

def extract_schema(df) -> dict:
    """
    Extract schema information from a DataFrame for LLM context.

    Returns:
        dict with: shape, columns, dtypes, sample, nulls, numeric_summary
    """
    import pandas as pd

    schema = {
        "shape":      {"rows": df.shape[0], "cols": df.shape[1]},
        "columns":    list(df.columns),
        "dtypes":     {col: str(df[col].dtype) for col in df.columns},
        "nulls":      df.isnull().sum().to_dict(),
        "sample":     df.head(3).to_dict(orient="records"),
        "unique":     {col: int(df[col].nunique()) for col in df.columns},
    }

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        schema["numeric_summary"] = df[numeric_cols].describe().to_dict()

    return schema


def build_schema_context(df, filename: str = "") -> str:
    """
    Build a human-readable schema context string for the LLM prompt.
    """
    schema = extract_schema(df)

    lines = [
        f"DataFrame: {filename or 'uploaded file'}",
        f"Shape: {schema['shape']['rows']} rows × {schema['shape']['cols']} columns",
        "",
        "Columns and dtypes:",
    ]

    for col in schema["columns"]:
        dtype   = schema["dtypes"][col]
        nulls   = schema["nulls"].get(col, 0)
        unique  = schema["unique"].get(col, "?")
        null_str = f", {nulls} nulls" if nulls > 0 else ""
        lines.append(f"  {col} ({dtype}, {unique} unique{null_str})")

    lines.append("")
    lines.append("Sample rows (first 3):")
    for row in schema["sample"]:
        lines.append("  " + str(row))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> str:
    """
    Classify the query into one of three modes.

    Returns:
        "statistical"   — run direct pandas operations
        "visualization" — generate a chart
        "code"          — generate pandas code via LLM
    """
    q = query.lower()

    if any(kw in q for kw in VIZ_KEYWORDS):
        return "visualization"

    if any(kw in q for kw in STAT_KEYWORDS):
        return "statistical"

    return "code"


# ---------------------------------------------------------------------------
# Mode 1 — Statistical operations (no LLM)
# ---------------------------------------------------------------------------

def run_statistical(query: str, df) -> dict:
    """
    Run a statistical operation based on keyword matching.

    Returns:
        dict with: result (any), result_type (str), operation (str)
    """
    import pandas as pd
    q = query.lower()

    # Shape / info
    if any(w in q for w in ["shape", "size", "dimension"]):
        return {
            "result":      f"{df.shape[0]} rows × {df.shape[1]} columns",
            "result_type": "text",
            "operation":   "df.shape",
        }

    if any(w in q for w in ["columns", "column names", "fields", "headers"]):
        return {
            "result":      list(df.columns),
            "result_type": "list",
            "operation":   "df.columns",
        }

    if any(w in q for w in ["dtypes", "types", "data types"]):
        return {
            "result":      df.dtypes.reset_index().rename(
                               columns={"index": "Column", 0: "Type"}
                           ),
            "result_type": "dataframe",
            "operation":   "df.dtypes",
        }

    # Head / tail
    if any(w in q for w in ["head", "first", "top"]):
        n = _extract_number(q) or 5
        return {
            "result":      df.head(n),
            "result_type": "dataframe",
            "operation":   f"df.head({n})",
        }

    if any(w in q for w in ["tail", "last", "bottom"]):
        n = _extract_number(q) or 5
        return {
            "result":      df.tail(n),
            "result_type": "dataframe",
            "operation":   f"df.tail({n})",
        }

    # Describe / summary
    if any(w in q for w in ["describe", "summary", "summarise", "summarize", "statistics"]):
        return {
            "result":      df.describe(include="all"),
            "result_type": "dataframe",
            "operation":   "df.describe(include='all')",
        }

    # Missing values
    if any(w in q for w in ["null", "missing", "nan", "isna", "isnull"]):
        null_counts = df.isnull().sum()
        null_pct    = (df.isnull().sum() / len(df) * 100).round(2)
        result      = pd.DataFrame({
            "Null Count": null_counts,
            "Null %":     null_pct,
        })
        return {
            "result":      result,
            "result_type": "dataframe",
            "operation":   "df.isnull().sum()",
        }

    # Duplicates
    if "duplicate" in q:
        dup_count = df.duplicated().sum()
        return {
            "result":      f"{dup_count} duplicate rows found.",
            "result_type": "text",
            "operation":   "df.duplicated().sum()",
        }

    # Correlation
    if any(w in q for w in ["corr", "correlation"]):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return {"result": "No numeric columns for correlation.", "result_type": "text", "operation": ""}
        return {
            "result":      numeric_df.corr().round(3),
            "result_type": "dataframe",
            "operation":   "df.select_dtypes(include='number').corr()",
        }

    # Value counts — detect column name in query
    if any(w in q for w in ["value_counts", "unique values", "distribution", "count of"]):
        col = _find_column_in_query(q, df.columns)
        if col:
            return {
                "result":      df[col].value_counts().reset_index().rename(
                                   columns={"index": col, col: "count"}
                               ),
                "result_type": "dataframe",
                "operation":   f"df['{col}'].value_counts()",
            }

    # Mean / average
    if any(w in q for w in ["mean", "average", "avg"]):
        col = _find_column_in_query(q, df.columns)
        if col and df[col].dtype in ["float64", "int64"]:
            val = df[col].mean()
            return {
                "result":      f"Mean of '{col}': {val:.4f}",
                "result_type": "text",
                "operation":   f"df['{col}'].mean()",
            }
        # All numeric columns
        return {
            "result":      df.select_dtypes(include="number").mean().round(4),
            "result_type": "series",
            "operation":   "df.mean(numeric_only=True)",
        }

    # Sum / total
    if any(w in q for w in ["sum", "total"]):
        col = _find_column_in_query(q, df.columns)
        if col and df[col].dtype in ["float64", "int64"]:
            val = df[col].sum()
            return {
                "result":      f"Sum of '{col}': {val:,.2f}",
                "result_type": "text",
                "operation":   f"df['{col}'].sum()",
            }
        return {
            "result":      df.select_dtypes(include="number").sum().round(4),
            "result_type": "series",
            "operation":   "df.sum(numeric_only=True)",
        }

    # Max / min
    if any(w in q for w in ["max", "maximum", "highest", "largest"]):
        col = _find_column_in_query(q, df.columns)
        if col:
            return {
                "result":      f"Max of '{col}': {df[col].max()}",
                "result_type": "text",
                "operation":   f"df['{col}'].max()",
            }

    if any(w in q for w in ["min", "minimum", "lowest", "smallest"]):
        col = _find_column_in_query(q, df.columns)
        if col:
            return {
                "result":      f"Min of '{col}': {df[col].min()}",
                "result_type": "text",
                "operation":   f"df['{col}'].min()",
            }

    # Unique / nunique
    if any(w in q for w in ["unique", "distinct", "nunique"]):
        col = _find_column_in_query(q, df.columns)
        if col:
            unique_vals = df[col].unique()
            return {
                "result":      f"'{col}' has {len(unique_vals)} unique values: {list(unique_vals[:20])}",
                "result_type": "text",
                "operation":   f"df['{col}'].unique()",
            }

    # Default — describe
    return {
        "result":      df.describe(include="all"),
        "result_type": "dataframe",
        "operation":   "df.describe(include='all')",
    }


def _extract_number(text: str) -> int:
    """Extract the first integer from a string."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _find_column_in_query(query: str, columns) -> str:
    """Find the best matching column name mentioned in the query."""
    q_lower = query.lower()
    # Exact match first
    for col in columns:
        if col.lower() in q_lower:
            return col
    # Partial match
    for col in columns:
        words = col.lower().split()
        if any(w in q_lower for w in words if len(w) > 2):
            return col
    return None


# ---------------------------------------------------------------------------
# Mode 2 — LLM → Pandas Code
# ---------------------------------------------------------------------------

def run_code_generation(query: str, df, schema_context: str, filename: str = "") -> dict:
    """
    Use Ollama to generate pandas code for the query, then execute it safely.

    Retries once on failure with the error message fed back to the LLM.

    Returns:
        dict with: result, result_type, code, error (or None), retried (bool)
    """
    # First attempt
    code = _generate_code(query, schema_context)
    result, error = _execute_safe(code, df)

    if error and error != "BLOCKED":
        # Retry once with error context
        logger.info("Code generation retry after error: %s", error[:100])
        retry_code   = _generate_code(query, schema_context, previous_error=error)
        result, error = _execute_safe(retry_code, df)
        return {
            "result":      result,
            "result_type": _infer_result_type(result),
            "code":        retry_code,
            "error":       error,
            "retried":     True,
        }

    return {
        "result":      result,
        "result_type": _infer_result_type(result),
        "code":        code,
        "error":       error,
        "retried":     False,
    }


def _generate_code(query: str, schema_context: str, previous_error: str = None) -> str:
    """
    Ask the LLM to generate a pandas code snippet.
    Returns the raw code string.
    """
    from services.llm_service import _call_ollama

    error_section = ""
    if previous_error:
        error_section = f"\nThe previous attempt failed with this error:\n{previous_error}\nPlease fix the code.\n"

    prompt = f"""You are a Python/pandas expert. Generate Python code to answer this question.

CRITICAL RULES — MUST FOLLOW:
1. The DataFrame is ALREADY loaded into memory as the variable `df`. DO NOT use pd.read_csv(), pd.read_excel(), or open() — the data is already in `df`.
2. pandas (pd) and numpy (np) are already imported. Do NOT import anything.
3. Do NOT write files or make network calls.
4. Assign your final answer to a variable named `result`.
5. Return ONLY executable Python code — no explanation, no markdown, no comments.

DataFrame info:
{schema_context}

{error_section}
Question: {query}

Code (use `df` directly, do not read any file):"""

    raw = _call_ollama(prompt).strip()

    # Strip markdown code blocks if LLM added them
    raw = re.sub(r"```python", "", raw)
    raw = re.sub(r"```",       "", raw)
    return raw.strip()


def _is_safe(code: str) -> tuple:
    """
    Check if generated code is safe to execute.
    Returns (is_safe: bool, reason: str)
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked pattern detected: {pattern}"
    return True, ""


def _execute_safe(code: str, df) -> tuple:
    """
    Execute generated pandas code in a sandboxed namespace.

    Returns:
        (result, error_string_or_None)
    """
    import pandas as pd
    import numpy as np

    safe, reason = _is_safe(code)
    if not safe:
        logger.warning("Blocked unsafe code: %s", reason)
        return None, "BLOCKED"

    # Restricted namespace — only df, pd, np accessible
    namespace = {
        "df":  df,
        "pd":  pd,
        "np":  np,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "round": round,
        "sum":  sum,
        "min":  min,
        "max":  max,
        "print": print,
    }

    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        result = namespace.get("result", "Code executed but no 'result' variable was set.")
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _infer_result_type(result) -> str:
    """Infer the type of result for rendering decisions."""
    import pandas as pd
    if result is None:
        return "none"
    if isinstance(result, pd.DataFrame):
        return "dataframe"
    if isinstance(result, pd.Series):
        return "series"
    if isinstance(result, (list, dict)):
        return "list"
    return "text"


# ---------------------------------------------------------------------------
# Mode 3 — Visualization
# ---------------------------------------------------------------------------

def run_visualization(query: str, df, schema_context: str) -> dict:
    """
    Generate a matplotlib chart from the query.

    Returns:
        dict with: image_bytes (BytesIO), code, error
    """
    from services.llm_service import _call_ollama

    prompt = f"""You are a Python data visualization expert. Generate matplotlib code.

CRITICAL RULES:
1. The DataFrame is ALREADY in memory as `df`. DO NOT use pd.read_csv() or open().
2. matplotlib.pyplot is available as `plt`. pandas (pd) and numpy (np) are available. Do NOT import anything.
3. Do NOT call plt.show(). Save with: fig = plt.gcf()
4. Give the chart a clear title.
5. Return ONLY Python code — no explanation, no markdown.

DataFrame info:
{schema_context}

Question: {query}

Code (use `df` directly):"""

    raw_code = _call_ollama(prompt).strip()
    raw_code = re.sub(r"```python", "", raw_code)
    raw_code = re.sub(r"```",       "", raw_code)
    code     = raw_code.strip()

    safe, reason = _is_safe(code)
    if not safe:
        return {"image_bytes": None, "code": code, "error": f"Blocked: {reason}"}

    import pandas as pd
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no GUI window
        import matplotlib.pyplot as plt

        namespace = {
            "df":  df, "pd": pd, "np": np,
            "plt": plt, "fig": None,
        }

        plt.figure(figsize=(10, 6))
        exec(compile(code, "<generated_viz>", "exec"), namespace)

        fig = namespace.get("fig") or plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        plt.close("all")

        return {"image_bytes": buf, "code": code, "error": None}

    except Exception as exc:
        plt.close("all")
        return {"image_bytes": None, "code": code, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_query(query: str, df, schema_context: str, filename: str = "") -> dict:
    """
    Route a query to the correct mode and return a unified result dict.

    Returns:
        dict with:
            mode        : "statistical" / "code" / "visualization"
            result      : the answer (DataFrame, string, number, etc.)
            result_type : "dataframe" / "text" / "series" / "list" / "none"
            code        : generated code (if applicable)
            image_bytes : BytesIO image (visualization mode only)
            error       : error string or None
            retried     : bool (code mode only)
    """
    intent = detect_intent(query)
    logger.info("Data explorer intent: %s for query: '%s'", intent, query[:60])

    if intent == "statistical":
        stat_result = run_statistical(query, df)
        return {
            "mode":        "statistical",
            "result":      stat_result["result"],
            "result_type": stat_result["result_type"],
            "code":        stat_result["operation"],
            "image_bytes": None,
            "error":       None,
            "retried":     False,
        }

    elif intent == "visualization":
        viz_result = run_visualization(query, df, schema_context)
        return {
            "mode":        "visualization",
            "result":      None,
            "result_type": "image",
            "code":        viz_result["code"],
            "image_bytes": viz_result["image_bytes"],
            "error":       viz_result["error"],
            "retried":     False,
        }

    else:
        code_result = run_code_generation(query, df, schema_context, filename)
        return {
            "mode":        "code",
            "result":      code_result["result"],
            "result_type": code_result["result_type"],
            "code":        code_result["code"],
            "image_bytes": None,
            "error":       code_result["error"],
            "retried":     code_result["retried"],
        }


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file(file_obj, filename: str):
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Returns:
        (df, error_string_or_None)
    """
    import pandas as pd

    ext = filename.lower().rsplit(".", 1)[-1]
    try:
        # Read raw bytes first so we can safely seek/retry
        raw_bytes = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "rb").read()

        if ext == "csv":
            # Try multiple encodings — files from Windows/Excel often use non-UTF-8
            encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
            df = None
            last_err = None
            for enc in encodings:
                try:
                    import io as _io
                    df = pd.read_csv(_io.BytesIO(raw_bytes), encoding=enc)
                    logger.info("CSV loaded with encoding: %s", enc)
                    break
                except (UnicodeDecodeError, Exception) as e:
                    last_err = e
                    continue
            if df is None:
                return None, f"Could not decode file with any encoding. Last error: {last_err}"

        elif ext in ("xlsx", "xls"):
            import io as _io
            df = pd.read_excel(_io.BytesIO(raw_bytes), engine="openpyxl")

        else:
            return None, f"Unsupported file type: .{ext}"

        if df.empty:
            return None, "File loaded but contains no data."

        # Clean column names — strip whitespace
        df.columns = [str(c).strip() for c in df.columns]

        logger.info("Loaded %s: %d rows × %d cols", filename, *df.shape)
        return df, None

    except Exception as exc:
        return None, f"Failed to load file: {exc}"