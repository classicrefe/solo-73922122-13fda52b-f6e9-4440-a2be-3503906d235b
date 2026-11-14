"""上传 -> 校验 -> 确认错误 -> 缩减 的完整流水线。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config
from .errors import ValidationError
from .jobs import Job
from .migration import check_queries, parse_query
from .reducer import reduce_database
from .runner import (
    ScriptResult,
    TrialTimeout,
    confirm_baseline,
    execute_migration,
    make_canonical_copy,
)
from .script_guard import parse_script
from .validation import validate_database


def run_pipeline(job: Job, uploaded_db: str, raw_script: bytes) -> None:
    job.status = "running"
    job.message = "正在校验数据库结构与外键…"

    script = parse_script(raw_script)
    schema = validate_database(uploaded_db)

    job_dir = Path(job.dir)
    canonical = job_dir / "canonical.db"
    make_canonical_copy(uploaded_db, str(canonical))

    job.message = "正在独立副本上确认原始错误并校验确定性…"
    baseline = confirm_baseline(str(canonical), script, str(job_dir))
    if baseline.ok:
        # 原脚本成功 -> 无需缩减
        job.status = "done"
        job.message = "原脚本执行成功，没有产生错误，因此无需缩减。"
        job.result = {
            "kind": "success_no_reduction",
            "baseline": baseline.to_dict(),
            "row_counts_before": {
                table: sum(1 for (t, _pk) in schema.rows if t == table)
                for table in schema.tables
            },
            "row_counts_after": {
                table: sum(1 for (t, _pk) in schema.rows if t == table)
                for table in schema.tables
            },
            "tables": schema.tables,
            "trials": 0,
            "download_available": False,
        }
        return

    job.message = "已复现原始错误，开始 Delta Debugging 缩减…"
    reduced = job_dir / "reduced.db"
    outcome = reduce_database(
        str(canonical), script, str(job_dir), schema, baseline, str(reduced)
    )

    job.status = "done"
    job.message = "缩减完成：删除任一行及其外键闭包后都无法再复现同一错误。"
    job.result = {
        "kind": "reduced",
        "baseline": outcome.baseline.to_dict(),
        "final": outcome.final.to_dict(),
        "row_counts_before": outcome.row_counts_before,
        "row_counts_after": outcome.row_counts_after,
        "tables": schema.tables,
        "trials": outcome.stats.trials,
        "accepted_steps": outcome.stats.accepted_steps,
        "kept_rows": [
            {"table": table, "pk": pk} for table, pk in sorted(outcome.kept_rows)
        ],
        "log": outcome.stats.log,
        "download_available": True,
    }


# ---------------------------------------------------------------------------
# 迁移取数核查流水线
# ---------------------------------------------------------------------------

_MIGRATION_MESSAGES = {
    "migration_failed": "迁移脚本未完整成功，不提供升级库。",
    "migration_invalid_schema": "迁移后的目标库未通过结构与外键校验，不提供升级库。",
    "query_failed": "取数查询执行失败，不提供升级库。",
    "query_rejected": "取数查询超出 200 行或试跑时限，不予放行。",
    "nondeterministic": "取数查询结果不具备确定性，不予放行。",
    "mismatch": "原库与目标库的取数结果不一致，不提供升级库。",
    "verified": "全部取数查询两侧一致，可下载升级库。",
}


def _finish_migration(job: Job, kind: str, migration: dict[str, object],
                      queries: list[dict[str, object]],
                      schema_error: str | None = None,
                      extra: dict[str, object] | None = None) -> None:
    job.status = "done"
    job.message = _MIGRATION_MESSAGES[kind]
    result: dict[str, object] = {
        "kind": kind,
        "migration": migration,
        "queries": queries,
        "download_available": kind == "verified",
    }
    if schema_error is not None:
        result["schema_error"] = schema_error
    if extra:
        result.update(extra)
    job.result = result


def run_migration_pipeline(job: Job, uploaded_db: str, raw_migration: bytes,
                           raw_queries: list[str]) -> None:
    """迁移取数核查：独立副本迁移 -> 校验目标库 -> 逐条比对取数结果。

    任何失败都不提供升级库；原件与已有缩减结果始终不被修改。
    """
    job.status = "running"
    job.message = "正在校验数据库结构与外键…"

    script = parse_script(raw_migration)
    queries = [parse_query(q, i) for i, q in enumerate(raw_queries, start=1)]
    schema = validate_database(uploaded_db)

    job_dir = Path(job.dir)
    canonical = job_dir / "canonical.db"
    make_canonical_copy(uploaded_db, str(canonical))

    job.message = "正在独立副本上执行迁移脚本…"
    upgraded = job_dir / "upgraded.db"
    try:
        migration = execute_migration(str(canonical), script, str(upgraded))
    except TrialTimeout as exc:
        migration = ScriptResult(
            ok=False, error_code=None, error_name="TRIAL_TIMEOUT",
            error_message=str(exc),
        )
    if not migration.ok:
        _finish_migration(job, "migration_failed", migration.to_dict(), [])
        return

    job.message = "迁移成功，正在按已提交数据校验目标库结构与外键…"
    try:
        new_schema = validate_database(str(upgraded))
        conn = sqlite3.connect(f"file:{upgraded}?mode=ro", uri=True)
        try:
            verdict = conn.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            conn.close()
        if verdict != "ok":
            raise ValidationError(f"目标库完整性检查失败：{verdict}")
    except ValidationError as exc:
        _finish_migration(
            job, "migration_invalid_schema", migration.to_dict(), [],
            schema_error=str(exc),
        )
        return

    job.message = "正在运行取数查询并逐条比对两侧结果…"
    kind, reports = check_queries(str(canonical), str(upgraded), queries)
    _finish_migration(
        job, kind, migration.to_dict(),
        [report.to_dict() for report in reports],
        extra={
            "tables_before": schema.tables,
            "tables_after": new_schema.tables,
            "row_limit": config.MAX_QUERY_ROWS,
        },
    )
