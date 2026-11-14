"""FastAPI 入口：上传、轮询任务、查看结果、下载复现库 / 升级库。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .errors import JobError, ScriptRejected, ValidationError
from .jobs import migration_registry, registry
from .samples import ensure_samples
from .service import run_migration_pipeline, run_pipeline

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_samples(STATIC_DIR)
    registry.cleanup_expired()
    migration_registry.cleanup_expired()

    async def _janitor() -> None:
        while True:
            await asyncio.sleep(60)
            registry.cleanup_expired()
            migration_registry.cleanup_expired()

    janitor = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        janitor.cancel()


app = FastAPI(title="SQLite 复现库缩减服务", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    return JSONResponse(ensure_samples(STATIC_DIR))


async def _read_db_upload(db: UploadFile) -> bytes:
    raw_db = await db.read()
    if len(raw_db) == 0:
        raise ValidationError("数据库文件为空。")
    if len(raw_db) > config.MAX_DB_BYTES:
        raise ValidationError(
            f"数据库大小 {len(raw_db)} 字节，超过上限 {config.MAX_DB_BYTES} 字节。"
        )
    if not raw_db.startswith(b"SQLite format 3\x00"):
        raise ValidationError("文件不是合法的 SQLite 数据库（魔数校验失败）。")
    return raw_db


def _save_upload(job, raw_db: bytes) -> str:
    upload_path = job.dir / "upload.db"
    upload_path.write_bytes(raw_db)
    # 原件落盘后只读保存，任何试跑都只接触副本
    Path(upload_path).chmod(0o444)
    return str(upload_path)


@app.post("/api/jobs")
async def create_job(
    db: UploadFile = File(..., description="满足约束的 SQLite 数据库"),
    sql: UploadFile | None = File(None, description="确定性 SQL 脚本文件"),
    script_text: str = Form("", description="或直接粘贴 SQL 脚本"),
) -> JSONResponse:
    raw_db = await _read_db_upload(db)

    if sql is not None:
        raw_script = await sql.read()
    else:
        raw_script = script_text.encode("utf-8")
    if not raw_script.strip():
        raise ValidationError("请上传或粘贴 SQL 脚本。")

    job = registry.create()
    upload_path = _save_upload(job, raw_db)
    registry.run_in_background(job, run_pipeline, upload_path, raw_script)
    return JSONResponse({"id": job.id, "status": job.status}, status_code=202)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = registry.get(job_id)
    return JSONResponse(job.public_dict())


@app.get("/api/jobs/{job_id}/download")
def download_reduced(job_id: str) -> FileResponse:
    job = registry.get(job_id)
    if job.status != "done" or not job.result.get("download_available"):
        raise JobError("缩减结果尚不可下载。")
    reduced = job.dir / "reduced.db"
    if not reduced.exists():
        raise JobError("缩减库文件已被清理。")
    return FileResponse(
        reduced,
        media_type="application/x-sqlite3",
        filename="repro.db",
    )


@app.post("/api/migration-jobs")
async def create_migration_job(
    db: UploadFile = File(..., description="满足原有约束的 SQLite 数据库"),
    migration: UploadFile | None = File(None, description="迁移 SQL 脚本文件"),
    migration_text: str = Form("", description="或直接粘贴迁移 SQL"),
    query1: str = Form("", description="取数查询 1（必填，只读 SELECT）"),
    query2: str = Form("", description="取数查询 2（可选）"),
    query3: str = Form("", description="取数查询 3（可选）"),
) -> JSONResponse:
    raw_db = await _read_db_upload(db)

    if migration is not None:
        raw_script = await migration.read()
    else:
        raw_script = migration_text.encode("utf-8")
    if not raw_script.strip():
        raise ValidationError("请上传或粘贴迁移 SQL。")

    queries = [q for q in (query1, query2, query3) if q and q.strip()]
    if not queries:
        raise ValidationError("请至少填写 1 条取数查询（只读 SELECT）。")
    if len(queries) > config.MAX_QUERIES:
        raise ValidationError(f"取数查询最多 {config.MAX_QUERIES} 条。")

    job = migration_registry.create()
    upload_path = _save_upload(job, raw_db)
    migration_registry.run_in_background(
        job, run_migration_pipeline, upload_path, raw_script, queries
    )
    return JSONResponse({"id": job.id, "status": job.status}, status_code=202)


@app.get("/api/migration-jobs/{job_id}")
def get_migration_job(job_id: str) -> JSONResponse:
    job = migration_registry.get(job_id)
    return JSONResponse(job.public_dict())


@app.get("/api/migration-jobs/{job_id}/download")
def download_upgraded(job_id: str) -> FileResponse:
    job = migration_registry.get(job_id)
    if job.status != "done" or not job.result.get("download_available"):
        raise JobError("升级库尚不可下载。")
    upgraded = job.dir / "upgraded.db"
    if not upgraded.exists():
        raise JobError("升级库文件已被清理。")
    return FileResponse(
        upgraded,
        media_type="application/x-sqlite3",
        filename="upgraded.db",
    )


@app.exception_handler(ValidationError)
@app.exception_handler(ScriptRejected)
@app.exception_handler(JobError)
def _handle_bad_request(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
