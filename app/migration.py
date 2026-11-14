"""迁移取数核查：只读查询解析、受限执行与两侧结果比对。

流程（由 service.run_migration_pipeline 驱动）：
1. 迁移脚本在独立副本上完整执行成功（见 runner.execute_migration）；
2. 连接关闭后按实际已提交的数据校验目标库结构与外键；
3. 每条取数查询（1~3 条确定性只读 SELECT）在原库与目标库上各跑两遍
   （两遍结果一致才认可其确定性），再逐条比对两侧结果：
   列名序列、每个值的 SQLite 类型与内容、重复行次数，忽略行顺序；
4. 查询结果超过 MAX_QUERY_ROWS 行或超过现有试跑时限一律不放行。
"""
from __future__ import annotations

import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field

from . import config
from .errors import ScriptRejected
from .script_guard import check_text_safety

# SQLite 存储类（typeof() 的五种取值）
Cell = tuple[str, object]  # (类型名, 值)
Row = tuple[Cell, ...]


class QueryRejected(Exception):
    """查询超出 200 行或试跑时限，不予放行。"""


class NonDeterministicQuery(Exception):
    """同一查询两次执行结果不一致，无法保证确定性。"""


# ---------------------------------------------------------------------------
# 查询文本静态检查
# ---------------------------------------------------------------------------

def _skip_ws_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
        else:
            break
    return i


def _first_keyword(text: str) -> str:
    i = _skip_ws_comments(text, 0)
    match = re.match(r"[A-Za-z_]+", text[i:])
    return match.group(0).upper() if match else ""


def _skip_quoted(text: str, i: int, quote: str) -> int:
    """跳过引号串（SQLite 用两个连续引号转义），返回串后位置。"""
    n = len(text)
    i += 1
    while i < n:
        if text[i] == quote:
            if i + 1 < n and text[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _ensure_single_statement(text: str) -> None:
    """只允许一条语句：第一个顶层分号之后只能是空白或注释。"""
    n = len(text)
    i = 0
    first_semi = -1
    while i < n:
        ch = text[i]
        if ch in ("'", '"', "`"):
            i = _skip_quoted(text, i, ch)
        elif ch == "[":
            j = text.find("]", i + 1)
            i = n if j == -1 else j + 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
        elif ch == ";":
            first_semi = i
            break
        else:
            i += 1
    if first_semi == -1:
        return
    rest = text[first_semi + 1:]
    if _skip_ws_comments(rest, 0) != len(rest):
        raise ScriptRejected("每条取数查询只能是单条 SELECT 语句。")


def parse_query(raw: str, index: int) -> str:
    """校验一条取数查询：非空、单条、只读 SELECT、不访问外部文件。"""
    if not raw or not raw.strip():
        raise ScriptRejected(f"第 {index} 条取数查询为空。")
    if "\x00" in raw:
        raise ScriptRejected(f"第 {index} 条取数查询包含 NUL 字节。")
    if len(raw.encode("utf-8")) > config.MAX_SCRIPT_BYTES:
        raise ScriptRejected(
            f"第 {index} 条取数查询超过 {config.MAX_SCRIPT_BYTES} 字节上限。"
        )
    text = raw.strip()
    check_text_safety(text)
    _ensure_single_statement(text)
    keyword = _first_keyword(text)
    if keyword not in ("SELECT", "WITH"):
        raise ScriptRejected(
            f"第 {index} 条取数查询必须是只读 SELECT（含 WITH...SELECT），"
            f"实际为 {keyword or '无法识别'}。"
        )
    return text


# ---------------------------------------------------------------------------
# 只读查询执行（行数上限 + 试跑时限 + authorizer）
# ---------------------------------------------------------------------------

@dataclass
class SelectResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    error_code: int | None = None
    error_name: str | None = None
    error_message: str | None = None

    def error_dict(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "error_name": self.error_name,
            "error_message": self.error_message,
        }


# 只读查询允许的最小动作集合；其余一律拒绝
_SELECT_ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
)


def _select_authorizer(action: int, arg1: str | None, arg2: str | None,
                       db_name: str | None, trigger: str | None) -> int:
    if action in _SELECT_ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and str(arg1).lower() not in (
        "load_extension",
        "readfile",
        "writefile",
    ):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _typed_cell(value: object) -> Cell:
    if value is None:
        return ("NULL", None)
    if isinstance(value, bytes):
        return ("BLOB", value)
    if isinstance(value, str):
        return ("TEXT", value)
    if isinstance(value, bool):  # 兜底：sqlite3 不会返回 bool
        return ("INTEGER", int(value))
    if isinstance(value, int):
        return ("INTEGER", value)
    if isinstance(value, float):
        return ("REAL", value)
    return ("TEXT", str(value))


def run_select(db_path: str, query: str) -> SelectResult:
    """以只读连接执行单条 SELECT，限制返回行数与试跑时限。"""
    result_box: dict[str, SelectResult] = {}
    error_box: dict[str, BaseException] = {}

    def _worker() -> None:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            conn.set_authorizer(_select_authorizer)
            try:
                cur = conn.execute(query)
                columns = [d[0] for d in cur.description or []]
                fetched = cur.fetchmany(config.MAX_QUERY_ROWS + 1)
                if len(fetched) > config.MAX_QUERY_ROWS:
                    error_box["too_many"] = QueryRejected(
                        f"查询结果超过 {config.MAX_QUERY_ROWS} 行上限，不予放行。"
                    )
                    return
                result_box["result"] = SelectResult(
                    ok=True,
                    columns=columns,
                    rows=[tuple(_typed_cell(v) for v in row) for row in fetched],
                )
            except sqlite3.Error as exc:
                result_box["result"] = SelectResult(
                    ok=False,
                    error_code=getattr(exc, "sqlite_errorcode", None),
                    error_name=getattr(exc, "sqlite_errorname", None),
                    error_message=str(exc),
                )
        except BaseException as exc:  # noqa: BLE001 - 回传线程内任何异常
            error_box["fatal"] = exc
        finally:
            if conn is not None:
                conn.close()

    thread = threading.Thread(target=_worker, name="sqlite-select", daemon=True)
    thread.start()
    thread.join(config.TRIAL_TIMEOUT_SECONDS)
    if thread.is_alive():
        raise QueryRejected(
            f"查询超过 {config.TRIAL_TIMEOUT_SECONDS} 秒试跑时限，不予放行。"
        )
    if "too_many" in error_box:
        raise error_box["too_many"]
    if "fatal" in error_box:
        raise error_box["fatal"]
    return result_box["result"]


# ---------------------------------------------------------------------------
# 结果比对（列名 + 值类型/内容 + 重复行次数，忽略行顺序）
# ---------------------------------------------------------------------------

def _format_cell(cell: Cell) -> str:
    type_name, value = cell
    if type_name == "NULL":
        return "NULL"
    if type_name == "BLOB":
        data = value if isinstance(value, bytes) else bytes(value)
        suffix = "…" if len(data) > 16 else ""
        return f"X'{data[:16].hex()}{suffix}'::BLOB"
    return f"{value!r}::{type_name}"


def _format_row(row: Row) -> str:
    return "(" + ", ".join(_format_cell(cell) for cell in row) + ")"


def compare_selects(original: SelectResult, migrated: SelectResult) -> list[str]:
    """逐条比对两侧查询结果，返回差异描述（空列表表示完全一致）。"""
    diffs: list[str] = []
    if original.columns != migrated.columns:
        diffs.append(
            f"列名不一致：原库 {original.columns}，目标库 {migrated.columns}"
        )
    before = Counter(original.rows)
    after = Counter(migrated.rows)
    for row, count in sorted((before - after).items(), key=lambda kv: repr(kv[0])):
        diffs.append(f"目标库缺少 {count} 行：{_format_row(row)}")
    for row, count in sorted((after - before).items(), key=lambda kv: repr(kv[0])):
        diffs.append(f"目标库多出 {count} 行：{_format_row(row)}")
    if len(diffs) > 50:
        return diffs[:50] + [f"……其余 {len(diffs) - 50} 处差异从略"]
    return diffs


# ---------------------------------------------------------------------------
# 逐条查询核查
# ---------------------------------------------------------------------------

def _cell_json(cell: Cell) -> dict[str, object]:
    type_name, value = cell
    if type_name == "NULL":
        return {"t": "NULL", "v": None}
    if type_name == "BLOB":
        data = value if isinstance(value, bytes) else bytes(value)
        suffix = f"…(共{len(data)}字节)" if len(data) > 64 else ""
        return {"t": "BLOB", "v": f"X'{data[:64].hex()}'{suffix}"}
    if type_name == "REAL":
        return {"t": "REAL", "v": repr(value)}
    return {"t": type_name, "v": value}


def _rows_json(rows: list[Row]) -> list[list[dict[str, object]]]:
    return [[_cell_json(cell) for cell in row] for row in rows]


@dataclass
class QueryReport:
    sql: str
    status: str = "ok"  # ok / mismatch / error / rejected / nondeterministic
    detail: str = ""
    error: dict[str, object] | None = None
    columns: list[str] | None = None
    original_count: int | None = None
    migrated_count: int | None = None
    original_rows: list[list[dict[str, object]]] | None = None
    migrated_rows: list[list[dict[str, object]]] | None = None
    differences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sql": self.sql,
            "status": self.status,
            "detail": self.detail,
            "error": self.error,
            "columns": self.columns,
            "original_count": self.original_count,
            "migrated_count": self.migrated_count,
            "original_rows": self.original_rows,
            "migrated_rows": self.migrated_rows,
            "differences": self.differences,
        }


def _run_deterministic(db_path: str, query: str) -> SelectResult:
    """同一查询执行两遍，结果一致才认可（确定性检查）。"""
    first = run_select(db_path, query)
    if not first.ok:
        return first
    second = run_select(db_path, query)
    if not second.ok:
        return second
    if first.columns != second.columns or Counter(first.rows) != Counter(second.rows):
        raise NonDeterministicQuery(
            "同一查询两次执行结果不一致，无法保证确定性，不予放行。"
        )
    return first


def _check_one_query(canonical_path: str, upgraded_path: str,
                     query: str) -> QueryReport:
    report = QueryReport(sql=query)
    try:
        original = _run_deterministic(canonical_path, query)
    except (QueryRejected, NonDeterministicQuery) as exc:
        report.status = (
            "rejected" if isinstance(exc, QueryRejected) else "nondeterministic"
        )
        report.detail = str(exc)
        return report
    if not original.ok:
        report.status = "error"
        report.detail = "查询在原库上执行失败。"
        report.error = original.error_dict()
        return report

    try:
        migrated = _run_deterministic(upgraded_path, query)
    except (QueryRejected, NonDeterministicQuery) as exc:
        report.status = (
            "rejected" if isinstance(exc, QueryRejected) else "nondeterministic"
        )
        report.detail = str(exc)
        return report
    if not migrated.ok:
        report.status = "error"
        report.detail = "查询在迁移后的目标库上执行失败。"
        report.error = migrated.error_dict()
        return report

    report.columns = original.columns
    report.original_count = len(original.rows)
    report.migrated_count = len(migrated.rows)
    diffs = compare_selects(original, migrated)
    if diffs:
        report.status = "mismatch"
        report.detail = f"两侧结果不一致，共 {len(diffs)} 处差异。"
        report.differences = diffs
        # 有差异时展示两侧完整结果（行数已被 200 行上限约束）
        report.original_rows = _rows_json(original.rows)
        report.migrated_rows = _rows_json(migrated.rows)
    else:
        report.status = "ok"
        report.detail = f"两侧一致（{len(original.rows)} 行）。"
    return report


_STATUS_TO_KIND = {
    "mismatch": "mismatch",
    "error": "query_failed",
    "rejected": "query_rejected",
    "nondeterministic": "nondeterministic",
}


def check_queries(canonical_path: str, upgraded_path: str,
                  queries: list[str]) -> tuple[str, list[QueryReport]]:
    """逐条核查取数查询，返回（整体结论 kind, 每条报告）。"""
    reports: list[QueryReport] = []
    overall = "verified"
    for query in queries:
        report = _check_one_query(canonical_path, upgraded_path, query)
        reports.append(report)
        if report.status != "ok" and overall == "verified":
            overall = _STATUS_TO_KIND[report.status]
    return overall, reports
