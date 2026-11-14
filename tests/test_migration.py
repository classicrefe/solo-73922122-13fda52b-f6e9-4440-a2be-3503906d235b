"""迁移取数核查：查询解析、只读执行、结果比对与端到端流水线。"""
from __future__ import annotations

import os
import sqlite3
import time
from types import SimpleNamespace

import pytest

# 每个测试进程使用独立 DATA_DIR
os.environ.setdefault("DATA_DIR", "/tmp/reducer-test-data")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.errors import ScriptRejected  # noqa: E402
from app.main import app  # noqa: E402
from app.migration import (  # noqa: E402
    QueryRejected,
    SelectResult,
    check_queries,
    compare_selects,
    parse_query,
    run_select,
)
from app.runner import TrialTimeout, execute_migration, make_canonical_copy  # noqa: E402
from app.service import run_migration_pipeline  # noqa: E402
from app.validation import validate_database  # noqa: E402


@pytest.fixture()
def payload_db(tmp_path):
    """含 BLOB 亲和列与重复值的库，用于类型/重复行比对。"""
    path = tmp_path / "payload.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE grp (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE item (
            id INTEGER PRIMARY KEY,
            payload BLOB,
            grp_id INTEGER REFERENCES grp(id)
        );
        INSERT INTO grp VALUES (1, 'g1'), (2, 'g2');
        INSERT INTO item VALUES (1, 5, 1), (2, 5, 1), (3, 7, 2);
        """
    )
    conn.commit()
    conn.close()
    return path


def _job(tmp_path):
    return SimpleNamespace(id="t", dir=tmp_path, status="", message="", result={})


def _run_pipeline(tmp_path, source_db, migration, queries):
    upload = tmp_path / "upload.db"
    upload.write_bytes(source_db.read_bytes())
    before = upload.read_bytes()
    job = _job(tmp_path)
    run_migration_pipeline(job, str(upload), migration.encode(), queries)
    assert upload.read_bytes() == before, "原件不得被修改"
    return job


# ---------------------------------------------------------------------------
# parse_query 静态检查
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "  -- 注释\nSELECT 1;  -- 尾部",
        "SELECT ';'",
        "WITH c AS (SELECT 1) SELECT * FROM c;",
        "SELECT 1; /* 收尾注释 */",
    ],
)
def test_parse_query_accepts(sql):
    assert parse_query(sql, 1)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "INSERT INTO t VALUES (1)",
        "DELETE FROM t",
        "SELECT 1; SELECT 2",
        "SELECT 1;;",
        "SELECT 1; DROP TABLE t",
        "ATTACH '/tmp/x.db' AS x;",
        "SELECT readfile('/etc/passwd');",
        "PRAGMA foreign_keys=ON;",
    ],
)
def test_parse_query_rejects(sql):
    with pytest.raises(ScriptRejected):
        parse_query(sql, 1)


def test_parse_query_rejects_nul():
    with pytest.raises(ScriptRejected):
        parse_query("SELECT 1;\x00", 2)


# ---------------------------------------------------------------------------
# run_select 只读执行
# ---------------------------------------------------------------------------

def test_run_select_returns_typed_cells(tmp_path):
    path = tmp_path / "types.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v BLOB)")
    conn.execute("INSERT INTO t VALUES (1, NULL)")
    conn.execute("INSERT INTO t VALUES (2, 42)")
    conn.execute("INSERT INTO t VALUES (3, 2.5)")
    conn.execute("INSERT INTO t VALUES (4, 'txt')")
    conn.execute("INSERT INTO t VALUES (5, X'ABCD')")
    conn.commit()
    conn.close()

    result = run_select(str(path), "SELECT v FROM t ORDER BY id")
    assert result.ok
    assert result.columns == ["v"]
    types = [row[0][0] for row in result.rows]
    assert types == ["NULL", "INTEGER", "REAL", "TEXT", "BLOB"]
    assert result.rows[1][0] == ("INTEGER", 42)
    assert result.rows[4][0] == ("BLOB", b"\xab\xcd")


def test_run_select_row_limit(tmp_path, cyclic_db):
    query = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
        "WHERE x < 500) SELECT x FROM c"
    )
    with pytest.raises(QueryRejected, match="200"):
        run_select(str(cyclic_db), query)


def test_run_select_timeout(monkeypatch, tmp_path, cyclic_db):
    monkeypatch.setattr(config, "TRIAL_TIMEOUT_SECONDS", 0)
    heavy = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
        "WHERE x < 5000000) SELECT SUM(x) FROM c"
    )
    with pytest.raises(QueryRejected, match="时限"):
        run_select(str(cyclic_db), heavy)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (SELECT 1) DELETE FROM dept",
        "DROP TABLE dept",
        "INSERT INTO dept VALUES (9, 'x')",
        "PRAGMA foreign_keys=ON",
        "ALTER TABLE dept ADD COLUMN x TEXT",
    ],
)
def test_run_select_authorizer_blocks_writes(tmp_path, cyclic_db, sql):
    result = run_select(str(cyclic_db), sql)
    assert not result.ok
    assert result.error_name == "SQLITE_AUTH"


# ---------------------------------------------------------------------------
# compare_selects 比对语义
# ---------------------------------------------------------------------------

def _result(columns, rows):
    return SelectResult(ok=True, columns=columns, rows=rows)


def test_compare_ignores_row_order():
    a = _result(["x"], [(("INTEGER", 1),), (("INTEGER", 2),)])
    b = _result(["x"], [(("INTEGER", 2),), (("INTEGER", 1),)])
    assert compare_selects(a, b) == []


def test_compare_detects_duplicate_counts():
    a = _result(["x"], [(("TEXT", "A"),), (("TEXT", "A"),), (("TEXT", "B"),)])
    b = _result(["x"], [(("TEXT", "A"),), (("TEXT", "B"),), (("TEXT", "B"),)])
    diffs = compare_selects(a, b)
    assert any("缺少 1 行" in d and "'A'" in d for d in diffs)
    assert any("多出 1 行" in d and "'B'" in d for d in diffs)


def test_compare_detects_type_change():
    a = _result(["v"], [(("INTEGER", 5),)])
    b = _result(["v"], [(("TEXT", "5"),)])
    diffs = compare_selects(a, b)
    assert any("INTEGER" in d for d in diffs)
    assert any("TEXT" in d for d in diffs)


def test_compare_detects_column_names():
    a = _result(["x", "y"], [])
    b = _result(["x", "z"], [])
    diffs = compare_selects(a, b)
    assert any("列名不一致" in d for d in diffs)


# ---------------------------------------------------------------------------
# execute_migration
# ---------------------------------------------------------------------------

def test_execute_migration_success(tmp_path, cyclic_db):
    canonical = tmp_path / "canon.db"
    make_canonical_copy(str(cyclic_db), str(canonical))
    dest = tmp_path / "up.db"
    result = execute_migration(
        str(canonical), "ALTER TABLE employee ADD COLUMN note TEXT;", str(dest)
    )
    assert result.ok
    cols = [r[1] for r in sqlite3.connect(str(dest)).execute(
        "PRAGMA table_info(employee)")]
    assert "note" in cols
    # 规范原件不受影响
    cols_old = [r[1] for r in sqlite3.connect(str(canonical)).execute(
        "PRAGMA table_info(employee)")]
    assert "note" not in cols_old


def test_execute_migration_failure_keeps_committed(tmp_path, cyclic_db):
    """先 COMMIT 再失败：失败返回错误，已提交部分留在目标库（按实际提交校验）。"""
    canonical = tmp_path / "canon.db"
    make_canonical_copy(str(cyclic_db), str(canonical))
    dest = tmp_path / "up.db"
    script = (
        "BEGIN; INSERT INTO dept (id, name) VALUES (9, 'New'); COMMIT; "
        "INSERT INTO nope VALUES (1);"
    )
    result = execute_migration(str(canonical), script, str(dest))
    assert not result.ok
    assert result.error_name == "SQLITE_ERROR"
    conn = sqlite3.connect(str(dest))
    assert conn.execute(
        "SELECT name FROM dept WHERE id = 9"
    ).fetchone()[0] == "New"
    conn.close()


def test_execute_migration_timeout(monkeypatch, tmp_path, cyclic_db):
    canonical = tmp_path / "canon.db"
    make_canonical_copy(str(cyclic_db), str(canonical))
    monkeypatch.setattr(config, "TRIAL_TIMEOUT_SECONDS", 0.2)
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
        "WHERE x < 3000000) SELECT SUM(x) FROM c;"
    )
    with pytest.raises(TrialTimeout):
        execute_migration(str(canonical), slow, str(tmp_path / "up.db"))


# ---------------------------------------------------------------------------
# 流水线（直接调用）
# ---------------------------------------------------------------------------

def test_pipeline_verified(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "ALTER TABLE employee ADD COLUMN note TEXT;",
        ["SELECT id, name FROM dept", "SELECT id, dept_id, mentor_id FROM employee"],
    )
    assert job.status == "done"
    assert job.result["kind"] == "verified"
    assert job.result["download_available"] is True
    upgraded = tmp_path / "upgraded.db"
    assert upgraded.exists()
    schema = validate_database(str(upgraded))
    assert set(schema.tables) == {"dept", "employee", "project"}
    assert sqlite3.connect(str(upgraded)).execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"


def test_pipeline_mismatch_value_change(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "UPDATE dept SET name = 'X' WHERE id = 1;",
        ["SELECT name FROM dept"],
    )
    assert job.result["kind"] == "mismatch"
    assert job.result["download_available"] is False
    report = job.result["queries"][0]
    assert report["status"] == "mismatch"
    assert report["original_rows"] and report["migrated_rows"]
    assert report["differences"]


def test_pipeline_mismatch_type_change(tmp_path, payload_db):
    job = _run_pipeline(
        tmp_path, payload_db,
        "UPDATE item SET payload = CAST(payload AS TEXT);",
        ["SELECT payload FROM item"],
    )
    assert job.result["kind"] == "mismatch"
    diffs = job.result["queries"][0]["differences"]
    assert any("INTEGER" in d for d in diffs)
    assert any("TEXT" in d for d in diffs)


def test_pipeline_mismatch_duplicate_count(tmp_path, payload_db):
    job = _run_pipeline(
        tmp_path, payload_db,
        "DELETE FROM item WHERE id = 2;",
        ["SELECT payload FROM item"],
    )
    assert job.result["kind"] == "mismatch"
    diffs = job.result["queries"][0]["differences"]
    assert any("缺少 1 行" in d for d in diffs)


def test_pipeline_ignores_row_order(tmp_path, payload_db):
    # 删除并重新插入 id=1：物理顺序改变，多重集不变
    job = _run_pipeline(
        tmp_path, payload_db,
        "DELETE FROM item WHERE id = 1; "
        "INSERT INTO item (id, payload, grp_id) VALUES (1, 5, 1);",
        ["SELECT payload FROM item"],
    )
    assert job.result["kind"] == "verified"


def test_pipeline_migration_failed(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db, "INSERT INTO nope VALUES (1);", ["SELECT 1"]
    )
    assert job.result["kind"] == "migration_failed"
    assert job.result["download_available"] is False
    assert job.result["migration"]["error_name"]


def test_pipeline_migration_invalid_schema(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "PRAGMA foreign_keys=OFF; DROP TABLE dept;",
        ["SELECT 1"],
    )
    assert job.result["kind"] == "migration_invalid_schema"
    assert job.result["schema_error"]
    assert job.result["download_available"] is False


def test_pipeline_query_failed(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "ALTER TABLE employee RENAME TO staff;",
        ["SELECT * FROM employee"],
    )
    assert job.result["kind"] == "query_failed"
    report = job.result["queries"][0]
    assert report["status"] == "error"
    assert report["error"]["error_message"]


def test_pipeline_query_rejected_too_many_rows(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "ALTER TABLE employee ADD COLUMN note TEXT;",
        ["WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
         "WHERE x < 500) SELECT x FROM c"],
    )
    assert job.result["kind"] == "query_rejected"
    assert job.result["download_available"] is False


def test_pipeline_nondeterministic_query(tmp_path, cyclic_db):
    job = _run_pipeline(
        tmp_path, cyclic_db,
        "ALTER TABLE employee ADD COLUMN note TEXT;",
        ["SELECT random()"],
    )
    assert job.result["kind"] == "nondeterministic"
    assert job.result["download_available"] is False


def test_check_queries_all_ok(tmp_path, cyclic_db):
    canonical = tmp_path / "canon.db"
    make_canonical_copy(str(cyclic_db), str(canonical))
    kind, reports = check_queries(
        str(canonical), str(canonical),
        ["SELECT * FROM dept", "SELECT id FROM employee"],
    )
    assert kind == "verified"
    assert [r.status for r in reports] == ["ok", "ok"]


# ---------------------------------------------------------------------------
# HTTP API 端到端
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        yield c


def _wait_done(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/migration-jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("任务超时未结束")


def _sample_db(client):
    return client.get("/static/sample_cyclic.db").content


def test_index_mentions_migration_check(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "迁移取数核查" in r.text


def test_api_verified_and_download(client, tmp_path):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={
            "migration_text": "ALTER TABLE employee ADD COLUMN note TEXT;",
            "query1": "SELECT id, name FROM dept",
            "query2": "SELECT id, dept_id, mentor_id FROM employee",
        },
    )
    assert r.status_code == 202, r.text
    job = _wait_done(client, r.json()["id"])
    assert job["status"] == "done"
    assert job["kind"] == "verified"
    assert job["download_available"] is True
    assert [q["status"] for q in job["queries"]] == ["ok", "ok"]

    dl = client.get(f"/api/migration-jobs/{job['id']}/download")
    assert dl.status_code == 200
    path = tmp_path / "upgraded.db"
    path.write_bytes(dl.content)
    schema = validate_database(str(path))
    assert set(schema.tables) == {"dept", "employee", "project"}
    cols = [r[1] for r in sqlite3.connect(str(path)).execute(
        "PRAGMA table_info(employee)")]
    assert "note" in cols


def test_api_mismatch_shows_both_sides_and_blocks_download(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={
            "migration_text": "UPDATE dept SET name = 'X' WHERE id = 1;",
            "query1": "SELECT name FROM dept",
        },
    )
    job = _wait_done(client, r.json()["id"])
    assert job["kind"] == "mismatch"
    assert job["download_available"] is False
    report = job["queries"][0]
    assert report["original_rows"] and report["migrated_rows"]
    assert report["differences"]
    dl = client.get(f"/api/migration-jobs/{job['id']}/download")
    assert dl.status_code == 400
    # 原件未被修改
    upload = config.DATA_DIR / "jobs" / job["id"] / "upload.db"
    assert upload.read_bytes() == _sample_db(client)


def test_api_migration_failed(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={"migration_text": "INSERT INTO nope VALUES (1);",
              "query1": "SELECT 1"},
    )
    job = _wait_done(client, r.json()["id"])
    assert job["kind"] == "migration_failed"
    assert job["migration"]["error_message"]
    assert client.get(f"/api/migration-jobs/{job['id']}/download").status_code == 400


def test_api_reject_attach_migration(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={"migration_text": "ATTACH '/tmp/x.db' AS x;", "query1": "SELECT 1"},
    )
    job = _wait_done(client, r.json()["id"])
    assert job["status"] == "error"
    assert "ATTACH" in job["message"]


def test_api_reject_non_select_query(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={"migration_text": "SELECT 1;", "query1": "DELETE FROM dept"},
    )
    job = _wait_done(client, r.json()["id"])
    assert job["status"] == "error"
    assert "SELECT" in job["message"]


def test_api_missing_query_400(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={"migration_text": "SELECT 1;"},
    )
    assert r.status_code == 400


def test_api_missing_migration_400(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("c.db", _sample_db(client), "application/x-sqlite3")},
        data={"query1": "SELECT 1"},
    )
    assert r.status_code == 400


def test_api_bad_db_400(client):
    r = client.post(
        "/api/migration-jobs",
        files={"db": ("bad.db", b"not sqlite", "application/octet-stream")},
        data={"migration_text": "SELECT 1;", "query1": "SELECT 1"},
    )
    assert r.status_code == 400


def test_api_missing_migration_job_400(client):
    assert client.get("/api/migration-jobs/nope").status_code == 400
