"""
SQL Agent — Plan & Execute pattern using SQLite.

Replaces the old BigQuery agent.  All SQL is generated against a local
SQLite database whose schema mirrors the original BigQuery tables.

Architecture (Plan & Execute — max 2 LLM calls per query)
----------------------------------------------------------
PHASE 1 - PLAN:    One LLM call to generate SQL (with schema context).
PHASE 2 - EXECUTE: Run SQL directly (zero LLM calls).
PHASE 3 - RECOVER: If phase 2 fails, ONE more LLM call to fix and retry.

Token savings vs ReAct: ~75 % (1-2 calls instead of 6-8).

Tools (available as standalone functions)
-----------------------------------------
get_schema(db_path, table)     → schema string
generate_sql(llm, q, schema)   → SQL string
validate_sql(sql)              → bool, error
execute_sql(db_path, sql)      → rows list
fix_sql(llm, sql, err, q, schema) → fixed SQL

Public API
----------
    agent = SQLAgent.from_config()   # or SQLAgent(db_path=..., llm=...)
    result = agent.run("What are the top suppliers by quality rating?")
    # result["final_answer"] / result["final_sql"] / result["rows"]
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.prompt_manager import SQL_GENERATE_PROMPT, SQL_FIX_PROMPT
from src.utils.custom_exception import SQLAgentException
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

HR_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "sqlite_agent" / "hr.db"
)


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE SCHEMA (static — used in prompts so LLM cannot hallucinate columns)
# ─────────────────────────────────────────────────────────────────────────────

SQLITE_SCHEMA = {
    "Departments": {
        "description": "Department info — budget, location, and designated manager.",
        "columns": {
            "DeptID":    "INTEGER PRIMARY KEY",
            "DeptName":  "TEXT — name of the department",
            "Location":  "TEXT — office city",
            "Budget":    "REAL — annual department budget in USD",
            "ManagerID": "INTEGER — FK → Employees.EmpID",
        },
    },
    "Jobs": {
        "description": "Job titles with salary bands, linked to departments.",
        "columns": {
            "JobID":      "INTEGER PRIMARY KEY",
            "JobTitle":   "TEXT — job title",
            "MinSalary":  "REAL — minimum salary for the role",
            "MaxSalary":  "REAL — maximum salary for the role",
            "DeptID":     "INTEGER — FK → Departments.DeptID",
        },
    },
    "Employees": {
        "description": "Core employee records. Central table — JOIN to all others via EmpID.",
        "columns": {
            "EmpID":     "INTEGER PRIMARY KEY",
            "FirstName": "TEXT",
            "LastName":  "TEXT",
            "Email":     "TEXT UNIQUE",
            "Phone":     "TEXT",
            "HireDate":  "TEXT (YYYY-MM-DD)",
            "JobID":     "INTEGER — FK → Jobs.JobID",
            "DeptID":    "INTEGER — FK → Departments.DeptID",
            "ManagerID": "INTEGER — FK → Employees.EmpID (self-referential)",
            "Status":    "TEXT — Active | Inactive",
        },
    },
    "Salaries": {
        "description": "Salary history per employee including bonus.",
        "columns": {
            "SalaryID":      "INTEGER PRIMARY KEY",
            "EmpID":         "INTEGER — FK → Employees.EmpID",
            "BaseSalary":    "REAL — annual base salary in USD",
            "Bonus":         "REAL — annual bonus amount",
            "EffectiveDate": "TEXT (YYYY-MM-DD)",
            "EndDate":       "TEXT (YYYY-MM-DD) — NULL means current",
        },
    },
    "Leaves": {
        "description": "Employee leave requests and approvals.",
        "columns": {
            "LeaveID":    "INTEGER PRIMARY KEY",
            "EmpID":      "INTEGER — FK → Employees.EmpID",
            "LeaveType":  "TEXT — Annual | Sick | Maternity | Paternity | Unpaid | Study",
            "StartDate":  "TEXT (YYYY-MM-DD)",
            "EndDate":    "TEXT (YYYY-MM-DD)",
            "Days":       "INTEGER — number of leave days",
            "Status":     "TEXT — Approved | Rejected | Pending",
            "ApprovedBy": "INTEGER — FK → Employees.EmpID",
        },
    },
    "PerformanceReviews": {
        "description": "Annual performance reviews with rating and reviewer.",
        "columns": {
            "ReviewID":   "INTEGER PRIMARY KEY",
            "EmpID":      "INTEGER — FK → Employees.EmpID",
            "ReviewYear": "INTEGER — year of review",
            "Rating":     "REAL — score 1.0 to 5.0",
            "ReviewerID": "INTEGER — FK → Employees.EmpID",
            "Goals":      "TEXT — goal set for this review period",
            "Comments":   "TEXT",
        },
    },
    "Training": {
        "description": "Employee course enrolments and completion scores.",
        "columns": {
            "TrainingID": "INTEGER PRIMARY KEY",
            "EmpID":      "INTEGER — FK → Employees.EmpID",
            "CourseName": "TEXT",
            "StartDate":  "TEXT (YYYY-MM-DD)",
            "EndDate":    "TEXT (YYYY-MM-DD)",
            "Status":     "TEXT — Enrolled | In Progress | Completed | Dropped",
            "Score":      "REAL — completion score (NULL if not completed)",
        },
    },
    "Attendance": {
        "description": "Daily attendance log per employee.",
        "columns": {
            "AttendanceID": "INTEGER PRIMARY KEY",
            "EmpID":        "INTEGER — FK → Employees.EmpID",
            "WorkDate":     "TEXT (YYYY-MM-DD)",
            "Status":       "TEXT — Present | Absent | WFH | Leave",
            "CheckIn":      "TEXT (HH:MM) — NULL if absent/leave",
            "CheckOut":     "TEXT (HH:MM) — NULL if absent/leave",
        },
    },
}


def _build_schema_text(schema: dict = SQLITE_SCHEMA) -> str:
    """Convert schema dict → human-readable text for LLM prompts."""
    lines: List[str] = []
    for table, info in schema.items():
        lines.append(f"Table: {table}  —  {info['description']}")
        for col, desc in info["columns"].items():
            lines.append(f"  {col}: {desc}")
        lines.append("")
    return "\n".join(lines)


_SCHEMA_TEXT = _build_schema_text()

# Column-name whitelist for validation
_VALID_COLUMNS: Dict[str, set] = {
    table: set(info["columns"].keys())
    for table, info in SQLITE_SCHEMA.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TOOL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_schema(db_path: str, table_name: Optional[str] = None) -> str:
    """Introspect the SQLite database and return a schema string.

    Parameters
    ----------
    db_path:    Path to the SQLite file.
    table_name: If provided, return only that table's schema.

    Returns
    -------
    Schema description string.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [r[0] for r in cursor.fetchall()]

        if not tables:
            conn.close()
            return "No tables found in database."

        target_tables = [table_name] if table_name and table_name in tables else tables
        lines: List[str] = []

        for t in target_tables:
            cursor.execute(f"PRAGMA table_info({t})")
            cols = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            count = cursor.fetchone()[0]
            lines.append(f"Table: {t}  ({count} rows)")
            for col in cols:
                pk = " PRIMARY KEY" if col["pk"] else ""
                lines.append(f"  {col['name']} ({col['type']}){pk}")
            lines.append("")

        conn.close()
        return "\n".join(lines)
    except Exception as exc:
        logger.error("get_schema failed: %s", exc)
        return f"Schema error: {exc}"


def generate_sql(llm: Any, question: str, schema_text: str = _SCHEMA_TEXT) -> str:
    """Use the LLM to generate a SQLite SELECT query.

    Parameters
    ----------
    llm:         LangChain-compatible LLM (must have `.invoke()`).
    question:    Natural language question.
    schema_text: Schema context injected into the prompt.

    Returns
    -------
    Raw SQL string (no markdown, no explanation).
    """
    response = llm.invoke(SQL_GENERATE_PROMPT.format(schema_text=schema_text, question=question))
    sql = response.content.strip() if hasattr(response, "content") else str(response).strip()
    # Strip markdown fences
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```\s*", "", sql)
    sql = sql.strip().rstrip(";").strip()
    return sql


def validate_sql(sql: str) -> tuple[bool, str]:
    """Light-weight SQL validator (SELECT-only, no destructive ops).

    Returns
    -------
    (is_valid: bool, message: str)
    """
    clean = sql.strip().upper()
    if not clean.startswith("SELECT"):
        return False, "Only SELECT statements are permitted."
    bad_kw = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
              "TRUNCATE", "ATTACH", "DETACH", "PRAGMA"]
    for kw in bad_kw:
        if re.search(rf"\b{kw}\b", clean):
            return False, f"Forbidden keyword: {kw}"
    return True, "OK"


def execute_sql(db_path: str, sql: str) -> Dict[str, Any]:
    """Execute *sql* against the SQLite file at *db_path*.

    Returns
    -------
    {"success": bool, "rows": list[dict], "count": int, "error": str|None}
    """
    ok, msg = validate_sql(sql)
    if not ok:
        return {"success": False, "rows": [], "count": 0, "error": msg}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        raw = cursor.fetchall()
        rows = [dict(r) for r in raw]
        conn.close()
        logger.info("execute_sql: %d rows returned", len(rows))
        return {"success": True, "rows": rows, "count": len(rows), "error": None}
    except Exception as exc:
        logger.error("execute_sql failed: %s", exc)
        return {"success": False, "rows": [], "count": 0, "error": str(exc)}


def fix_sql(
    llm: Any,
    failed_sql: str,
    error: str,
    question: str,
    schema_text: str = _SCHEMA_TEXT,
) -> str:
    """Ask the LLM to fix a broken SQL query.

    Parameters
    ----------
    llm:        LangChain-compatible LLM.
    failed_sql: The SQL that produced an error.
    error:      Error message from execute_sql.
    question:   Original natural language question.
    schema_text:Schema context.

    Returns
    -------
    Corrected SQL string.
    """
    response = llm.invoke(SQL_FIX_PROMPT.format(
        question=question,
        failed_sql=failed_sql,
        error=error,
        schema_text=schema_text,
    ))
    sql = response.content.strip() if hasattr(response, "content") else str(response).strip()
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```\s*", "", sql)
    sql = sql.strip().rstrip(";").strip()
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# PLAN & EXECUTE AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SQLAgent:
    """Plan & Execute SQL agent backed by SQLite.

    Lifecycle
    ---------
    1. PLAN   — one LLM call → SQL query
    2. EXECUTE— run SQL (no LLM)
    3. RECOVER— if error: one more LLM call → fixed SQL → retry once

    Parameters
    ----------
    db_path:     Path to the SQLite database file.
    llm:         LangChain-compatible LLM.
    max_attempts:Maximum total attempts (default 2 = plan + one fix).
    """

    def __init__(self, db_path: str, llm: Any, max_attempts: int = 2):
        self.db_path     = db_path
        self.llm         = llm
        self.max_attempts = max_attempts

        if not Path(db_path).exists():
            logger.warning("SQLite DB not found: %s — run create_dummy_data.py", db_path)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config=None, llm: Any = None) -> "SQLAgent":
        """Build from src.config.Config and an optional LLM.

        If *config* is None, ``load_config()`` is used.
        If *llm* is None, the GroqProvider from config is used.
        """
        try:
            from src.config import load_config
            cfg = config or load_config()
            sql_cfg = cfg.sql

            db_path = sql_cfg.get("sqlite_path", HR_DB_PATH)

            if llm is None:
                from src.llm import get_llm_provider
                llm = get_llm_provider(cfg.llm).get_langchain_llm()

            return cls(
                db_path=db_path,
                llm=llm,
                max_attempts=sql_cfg.get("max_attempts", 2),
            )
        except Exception as exc:
            raise SQLAgentException("SQLAgent.from_config failed", exc) from exc

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, question: str) -> Dict[str, Any]:
        """Run Plan & Execute agent for *question*.

        Returns
        -------
        {
          "question":     str,
          "final_answer": str,   natural language answer
          "final_sql":    str,   SQL that produced the answer
          "rows":         list,  raw result rows
          "attempts":     int,   1 or 2
          "success":      bool,
        }
        """
        logger.info("[SQL Agent] question: %s", question)

        schema_text = get_schema(self.db_path)
        attempts    = 0
        sql_query   = ""

        # ──── PHASE 1: PLAN ───────────────────────────────────────────────────
        try:
            sql_query = generate_sql(self.llm, question, schema_text)
            logger.info("[SQL Agent] Phase 1 SQL: %s", sql_query[:120])
        except Exception as exc:
            raise SQLAgentException(f"SQL generation failed for: {question}", exc) from exc

        attempts += 1

        # ──── PHASE 2: EXECUTE ────────────────────────────────────────────────
        result = execute_sql(self.db_path, sql_query)

        if result["success"]:
            logger.info("[SQL Agent] Phase 2 success — %d rows", result["count"])
            return {
                "question":     question,
                "final_answer": self._format_answer(question, result["rows"]),
                "final_sql":    sql_query,
                "rows":         result["rows"],
                "attempts":     attempts,
                "success":      True,
            }

        # ──── PHASE 3: RECOVER ────────────────────────────────────────────────
        if attempts >= self.max_attempts:
            logger.warning("[SQL Agent] Max attempts reached — returning error")
            return {
                "question":     question,
                "final_answer": f"Could not answer: {result['error']}",
                "final_sql":    sql_query,
                "rows":         [],
                "attempts":     attempts,
                "success":      False,
            }

        logger.info("[SQL Agent] Phase 3 — fixing error: %s", result["error"])
        try:
            fixed_sql = fix_sql(self.llm, sql_query, result["error"], question, schema_text)
            logger.info("[SQL Agent] Phase 3 fixed SQL: %s", fixed_sql[:120])
        except Exception as exc:
            logger.error("[SQL Agent] fix_sql LLM call failed: %s", exc)
            return {
                "question":     question,
                "final_answer": f"SQL fix failed: {exc}",
                "final_sql":    sql_query,
                "rows":         [],
                "attempts":     attempts,
                "success":      False,
            }

        attempts += 1
        result2 = execute_sql(self.db_path, fixed_sql)

        if result2["success"]:
            logger.info("[SQL Agent] Phase 3 success — %d rows", result2["count"])
            return {
                "question":     question,
                "final_answer": self._format_answer(question, result2["rows"]),
                "final_sql":    fixed_sql,
                "rows":         result2["rows"],
                "attempts":     attempts,
                "success":      True,
            }

        logger.warning("[SQL Agent] Recovery failed: %s", result2["error"])
        return {
            "question":     question,
            "final_answer": f"Query failed after {attempts} attempts: {result2['error']}",
            "final_sql":    fixed_sql,
            "rows":         [],
            "attempts":     attempts,
            "success":      False,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_answer(question: str, rows: List[dict]) -> str:
        """Convert row dicts to a readable natural language answer."""
        if not rows:
            return "No results found."

        if len(rows) == 1:
            vals = list(rows[0].values())
            if len(vals) == 1:
                return f"Answer: {vals[0]}"
            return "Result: " + ", ".join(f"{k}: {v}" for k, v in rows[0].items())

        header = list(rows[0].keys())
        out = [f"Found {len(rows)} result(s):"]
        for row in rows[:10]:          # cap display at 10
            out.append("  " + ", ".join(f"{k}: {v}" for k, v in row.items()))
        if len(rows) > 10:
            out.append(f"  ... and {len(rows) - 10} more.")
        return "\n".join(out)
