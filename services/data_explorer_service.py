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
    # Block file writing from visualizations
    r"\.savefig\s*\(", r"plt\.savefig\s*\(",
]

# Statistical intent keywords
STAT_KEYWORDS = {
    "describe", "summary", "summarise", "summarize", "info", "shape",
    "head", "tail", "first", "last", "columns", "dtypes", "types",
    "null", "missing", "nan", "isnull", "isna", "duplicate",
    "unique", "nunique", "count", "value_counts",
    "correlation", "corr", "mean", "median", "mode", "std",
    "variance", "min", "max", "range", "percentile", "quantile",
    # Search and presence queries handled statistically
    "is there", "is present", "find", "search", "contains", "contain",
    "filter", "where", "which rows", "list all", "show all",
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

def _extract_search_term(query: str, columns) -> str:
    """
    Extract the search term from a query like:
      "is 'Toyota' present?" -> "Toyota"
      "find rows where brand is Honda" -> "Honda"
      "search for INV-2024" -> "INV-2024"
    """
    import re
    q = query

    # Extract quoted strings first
    quoted = re.findall(r"([\'\"])(.*?)\1", q)
    if quoted:
        return quoted[0][1] if quoted else None

    # Remove column names and common filler words from query
    col_words = set()
    for col in columns:
        col_words.update(col.lower().split())

    stopwords = {
        "is", "are", "there", "present", "find", "search", "show", "me",
        "all", "rows", "where", "which", "has", "have", "contain", "contains",
        "the", "a", "an", "in", "of", "for", "any", "exists", "list",
        "filter", "with", "by", "from", "column",
    } | col_words

    tokens = [t for t in re.findall(r"[\w\-\.]+", q) if t.lower() not in stopwords and len(t) >= 2]

    return tokens[0] if tokens else None


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

    # Presence/search queries go to statistical (uses str.contains — no LLM needed)
    if any(phrase in q for phrase in ["is there", "is present", "are there",
                                       "exists", "search for", "find me",
                                       "show all rows", "filter by"]):
        return "statistical"

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

    # Search / is present / contains
    if any(w in q for w in ["is there", "is present", "find", "search",
                             "contain", "has", "exists", "show me all",
                             "filter", "where", "which rows", "list all"]):
        col  = _find_column_in_query(q, df.columns)
        term = _extract_search_term(q, df.columns)
        if term and col:
            mask   = df[col].astype(str).str.contains(term, case=False, na=False, regex=False)
            found  = df[mask]
            if found.empty:
                # Try regex as fallback
                try:
                    mask  = df[col].astype(str).str.contains(term, case=False, na=False, regex=True)
                    found = df[mask]
                except Exception:
                    pass
            return {
                "result":      found if not found.empty else f"No rows found matching '{term}' in '{col}'.",
                "result_type": "dataframe" if not found.empty else "text",
                "operation":   f"df[df['{col}'].str.contains('{term}', case=False)]",
            }
        elif term:
            # Search across all string columns
            mask = df.apply(lambda col: col.astype(str).str.contains(term, case=False, na=False, regex=False)).any(axis=1)
            found = df[mask]
            return {
                "result":      found if not found.empty else f"No rows found matching '{term}' across all columns.",
                "result_type": "dataframe" if not found.empty else "text",
                "operation":   f"df[df.apply(lambda c: c.astype(str).str.contains('{term}', case=False)).any(axis=1)]",
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
        retry_code    = _generate_code(query, schema_context, previous_error=error)
        result, error = _execute_safe(retry_code, df)

        if error:
            # Both attempts failed — fall back to statistical mode
            logger.warning("Both code gen attempts failed. Falling back to statistical mode.")
            try:
                stat = run_statistical(query, df)
                return {
                    "result":      stat["result"],
                    "result_type": stat["result_type"],
                    "code":        stat["operation"],
                    "error":       None,
                    "retried":     True,
                    "fallback":    True,
                    "friendly_error": _classify_error(error),
                }
            except Exception:
                pass

        return {
            "result":      result,
            "result_type": _infer_result_type(result),
            "code":        retry_code,
            "error":       _classify_error(error) if error else None,
            "retried":     True,
            "fallback":    False,
        }

    return {
        "result":      result,
        "result_type": _infer_result_type(result),
        "code":        code,
        "error":       _classify_error(error) if error else None,
        "retried":     False,
        "fallback":    False,
    }


def _generate_code(query: str, schema_context: str, previous_error: str = None) -> str:
    """
    Ask the LLM to generate a pandas code snippet.
    Returns the raw code string.
    """
    from services.llm_service import _call_ollama

    error_section = ""
    if previous_error:
        error_section = (
            f"\nIMPORTANT: The previous attempt failed with:\n"
            f"{previous_error}\n"
            f"Fix the code. Common mistakes to avoid:\n"
            f"- Do NOT use pd.read_csv() or open() — df is already loaded\n"
            f"- Do NOT use .loc[] with a Series result from idxmax() — use .iloc[int_index] instead\n"
            f"- Assign scalar results directly to `result`, not row selections\n"
        )

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



def _preprocess_code(code: str, df_columns: list) -> str:
    """
    Fix common LLM code generation mistakes before execution.
    Applied to every generated code block — no LLM call needed.

    Fixes:
      1. Remove stray import statements (pd/np already in namespace)
      2. Remove any df = pd.read_* lines (df already loaded)
      3. Fix .loc[series.idxmax()] → .iloc[series.idxmax()] pattern
      4. Remove print() calls (stdout not captured in sandbox)
      5. Ensure result is assigned even if LLM used return instead
      6. Strip markdown artifacts
    """
    import re

    # Strip markdown
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```",       "", code)
    code = code.strip()

    lines     = code.split("\n")
    cleaned   = []

    for line in lines:
        stripped = line.strip()

        # Remove import statements — everything is pre-imported
        if re.match(r"^import\s+|^from\s+\S+\s+import", stripped):
            continue

        # Remove df = pd.read_*(...) — df is already loaded
        if re.match(r"^df\s*=\s*pd\.read_", stripped):
            continue

        # Remove standalone print() calls
        if re.match(r"^print\s*\(", stripped):
            continue

        cleaned.append(line)

    code = "\n".join(cleaned).strip()

    # Fix .loc[expr.idxmax()] → safer .iloc[int(expr.idxmax())]
    code = re.sub(
        r"\.loc\[([^\]]+)\.idxmax\(\)\]",
        r".iloc[int(\1.idxmax())]",
        code,
    )
    code = re.sub(
        r"\.loc\[([^\]]+)\.idxmin\(\)\]",
        r".iloc[int(\1.idxmin())]",
        code,
    )

    # If LLM used "return X" instead of "result = X", convert it
    code = re.sub(r"^return\s+(.+)$", r"result = \1", code, flags=re.MULTILINE)

    # If no result assignment found, wrap last expression as result
    if "result" not in code:
        lines = [l for l in code.split("\n") if l.strip()]
        if lines:
            last = lines[-1].strip()
            # If last line looks like an expression (not assignment/if/for/etc)
            if not re.match(r"^(if|for|while|def|class|try|with|import)", last):
                lines[-1] = f"result = {last}"
                code = "\n".join(lines)

    return code


def _classify_error(error: str) -> str:
    """
    Convert a raw Python error into a user-friendly message.
    Never exposes raw tracebacks to the user.
    """
    e = error.lower()

    if "keyerror" in e:
        col = error.split('KeyError:')[-1].strip().strip('"\'')
        return (
            f"Column '{col}' was not found. "
            f"Please check the column name — it may have different capitalisation or spacing."
        )
    if "valueerror" in e and "could not convert" in e:
        return "A column contains mixed data types (text and numbers). Try specifying the column more precisely."
    if "typeerror" in e:
        return "Data type mismatch — the operation is not compatible with this column's data type."
    if "indexerror" in e or "index" in e and "out of bounds" in e:
        return "Row index out of range. The dataset may have fewer rows than expected."
    if "attributeerror" in e:
        return "The operation is not supported for this data type. Try a different approach."
    if "syntaxerror" in e:
        return "The generated code had a syntax error. Please rephrase your question more specifically."
    if "nameerror" in e:
        return "A variable or function was referenced that doesn't exist in the analysis context."
    if "zerodivisionerror" in e:
        return "Division by zero encountered. The column may contain zero values."
    if "blocked" in e:
        return "The query was blocked for security reasons. Please use a different approach."
    if "no such file" in e or "filenotfound" in e:
        return "The code tried to read a file. Use the pre-loaded DataFrame 'df' directly instead."
    if "no 'result' variable" in e:
        return "The analysis ran but produced no output. Try rephrasing your question."

    # Fallback — generic but not the raw traceback
    return "The analysis could not be completed. Try rephrasing your question or use a quick query chip below."

def _execute_safe(code: str, df) -> tuple:
    """
    Execute generated pandas code in a sandboxed namespace.
    Works on a COPY of df so the original is never mutated.
    Post-processes result to sanitise common LLM mistakes.

    Returns:
        (result, error_string_or_None)
    """
    import pandas as pd
    import numpy as np

    # Pre-process: fix common LLM mistakes before safety check
    code = _preprocess_code(code, list(df.columns))

    safe, reason = _is_safe(code)
    if not safe:
        logger.warning("Blocked unsafe code: %s", reason)
        return None, f"Unsafe code blocked: {reason}"

    # Work on a copy — never mutate the original DataFrame
    namespace = {
        "df":     df.copy(),
        "pd":     pd,
        "np":     np,
        "len":    len,
        "str":    str,
        "int":    int,
        "float":  float,
        "list":   list,
        "dict":   dict,
        "round":  round,
        "sum":    sum,
        "min":    min,
        "max":    max,
        "abs":    abs,
        "range":  range,
        "print":  print,
        "sorted": sorted,
        "zip":    zip,
        "enumerate": enumerate,
    }

    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        result = namespace.get("result", "Code ran but no 'result' variable was set.")
        result = _sanitise_result(result, df)
        return result, None
    except KeyError as exc:
        return None, (
            f"KeyError: {exc}. "
            "Hint: use .iloc[] for positional indexing or .at[] for label-based scalar access."
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _sanitise_result(result, df):
    """
    Fix common LLM output mistakes before displaying:
      - idxmax/idxmin returning a Series used as .loc index → resolve to row
      - Single-column DataFrame → convert to Series for cleaner display
      - Huge results → truncate with message
    """
    import pandas as pd

    if result is None:
        return result

    # If result is a Series returned by idxmax/idxmin, fetch those rows
    if isinstance(result, pd.Series) and result.dtype == object:
        # Could be index labels — try to use as .loc safely
        try:
            candidate = df.loc[result]
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                return candidate
        except Exception:
            pass

    # If scalar idxmax result accidentally stored (int/str label)
    if isinstance(result, (int, str)):
        try:
            candidate = df.loc[[result]]
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                return candidate
        except Exception:
            pass

    # Truncate very large DataFrames for display
    if isinstance(result, pd.DataFrame) and len(result) > 500:
        trunc = result.head(500)
        trunc.attrs["truncated"] = f"Showing 500 of {len(result)} rows."
        return trunc

    return result


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


def get_result_notice(result) -> str:
    """Return a notice string if the result was truncated, else empty string."""
    import pandas as pd
    if isinstance(result, pd.DataFrame):
        return result.attrs.get("truncated", "")
    return ""


# ---------------------------------------------------------------------------
# Mode 3 — Visualization
# ---------------------------------------------------------------------------


def _preprocess_viz_code(code: str) -> str:
    """
    Fix common LLM mistakes in matplotlib/visualization code:

    1. Remove fig.savefig() / plt.savefig() — we capture via BytesIO
    2. Move fig = plt.gcf() to AFTER the plot commands — not before
    3. Fix plt.pie(value_counts_result, labels=unique_result) mismatch:
         value_counts() and unique() return different orderings.
         Replace with consistent: vc = df[col].value_counts()
                                   plt.pie(vc.values, labels=vc.index)
    4. Remove plt.show() — non-interactive backend
    5. Remove stray import statements
    """
    import re

    # Strip markdown
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```",       "", code)

    lines   = code.strip().split("\n")
    cleaned = []
    has_gcf = False

    for line in lines:
        stripped = line.strip()

        # Remove savefig calls
        if re.search(r"(plt\.savefig|fig\.savefig)\s*\(", stripped):
            continue

        # Remove plt.show()
        if stripped == "plt.show()":
            continue

        # Remove imports
        if re.match(r"^import\s+|^from\s+\S+\s+import", stripped):
            continue

        # Remove df = pd.read_*
        if re.match(r"^df\s*=\s*pd\.read_", stripped):
            continue

        # Track gcf assignment — we'll add it at the end
        if re.match(r"^fig\s*=\s*plt\.gcf\(\)", stripped):
            has_gcf = True
            continue   # don't add it here — add at end

        cleaned.append(line)

    # Fix value_counts + unique mismatch in plt.pie
    code = "\n".join(cleaned)
    code = _fix_pie_labels(code)

    # Always put fig = plt.gcf() at the very end
    code = code.rstrip()
    code += "\nfig = plt.gcf()"

    return code


def _fix_pie_labels(code: str) -> str:
    """
    Fix the common plt.pie mismatch:
      plt.pie(df['col'].value_counts(), labels=df['col'].unique())
    →
      _vc = df['col'].value_counts()
      plt.pie(_vc.values, labels=_vc.index, ...)

    This ensures labels always match the data order.
    """
    import re

    def replace_pie(match):
        full     = match.group(0)
        # Extract the series expression from value_counts call
        vc_match = re.search(r"df\[([^\]]+)\]\.value_counts\(\)", full)
        if not vc_match:
            return full

        col_expr = vc_match.group(1)  # e.g. 'Drivetrain'

        # Build a safe replacement
        setup  = f"_vc = df[{col_expr}].value_counts(dropna=True)"
        # Reconstruct pie call replacing the data and labels args
        new_pie = re.sub(
            r"df\[([^\]]+)\]\.value_counts\(\)",
            "_vc.values",
            full
        )
        # Replace any labels=df[...].unique() with labels=_vc.index
        new_pie = re.sub(
            r"labels\s*=\s*df\[([^\]]+)\]\.(unique|value_counts)\(\)(\.index)?",
            "labels=_vc.index",
            new_pie
        )
        return setup + "\n" + new_pie

    # Match plt.pie(...) calls that span one line
    code = re.sub(
        r"plt\.pie\([^)]+\.value_counts\(\)[^)]*\)",
        replace_pie,
        code,
        flags=re.DOTALL
    )
    return code

def run_visualization(query: str, df, schema_context: str) -> dict:
    """
    Generate a matplotlib chart from the query.

    Returns:
        dict with: image_bytes (BytesIO), code, error
    """
    from services.llm_service import _call_ollama

    prompt = f"""You are a Python data visualization expert. Generate matplotlib code.

CRITICAL RULES — MUST FOLLOW EXACTLY:
1. `df` is already in memory. DO NOT use pd.read_csv(), pd.read_excel(), or open().
2. `plt`, `pd`, `np` are already imported. DO NOT import anything.
3. Do NOT call plt.show() or fig.savefig() — output is captured automatically.
4. Put fig = plt.gcf() at the VERY LAST LINE — not before the plot.
5. Give the chart a clear title using plt.title().
6. Return ONLY Python code — no explanation, no markdown, no comments.

IMPORTANT — For pie charts:
  Use value_counts() for BOTH the data AND labels:
    vc = df['ColumnName'].value_counts(dropna=True)
    plt.pie(vc.values, labels=vc.index, autopct='%1.1f%%')
  NEVER mix value_counts() data with unique() labels — they have different ordering.

DataFrame info:
{schema_context}

Question: {query}

Code (use `df` directly, end with fig = plt.gcf()):"""

    raw_code = _call_ollama(prompt).strip()

    # Apply visualization-specific preprocessor before safety check
    code = _preprocess_viz_code(raw_code)

    safe, reason = _is_safe(code)
    if not safe:
        return {"image_bytes": None, "code": code, "error": f"Blocked: {reason}"}

    import pandas as pd
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        namespace = {
            "df":     df.copy(),   # never mutate original
            "pd":     pd,
            "np":     np,
            "plt":    plt,
            "fig":    None,
            "len":    len,
            "list":   list,
            "range":  range,
            "sorted": sorted,
            "str":    str,
            "int":    int,
            "float":  float,
        }

        plt.figure(figsize=(10, 6))
        plt.tight_layout()

        exec(compile(code, "<generated_viz>", "exec"), namespace)

        # Always grab the current figure — whether LLM set fig or not
        fig = namespace.get("fig") or plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        plt.close("all")

        return {"image_bytes": buf, "code": code, "error": None}

    except Exception as exc:
        plt.close("all")
        friendly = _classify_error(f"{type(exc).__name__}: {exc}")
        return {
            "image_bytes": None,
            "code":        code,
            "error":       friendly,
        }


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
            "mode":          "code" if not code_result.get("fallback") else "statistical",
            "result":        code_result["result"],
            "result_type":   code_result["result_type"],
            "code":          code_result["code"],
            "image_bytes":   None,
            "error":         code_result.get("error"),
            "friendly_error": code_result.get("friendly_error"),
            "retried":       code_result["retried"],
            "fallback":      code_result.get("fallback", False),
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