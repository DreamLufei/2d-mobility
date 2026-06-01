from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import WebConsoleSettings
from .models import (
    BatchJobRequest,
    ExternalEventResumeRequest,
    HitlResponseRequest,
    RuntimeSettingsUpdateRequest,
    SingleJobRequest,
    WikiQueryRequest,
    WikiReindexRequest,
)
from .service import WebConsoleService


def _service(request: Request) -> WebConsoleService:
    return request.app.state.console_service  # type: ignore[return-value]


def create_app(settings: WebConsoleSettings | None = None) -> FastAPI:
    resolved_settings = settings or WebConsoleSettings.from_repo()
    service = WebConsoleService(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.console_service = service
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="script_new Web Console", version="1.1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173", "http://localhost:4173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if resolved_settings.frontend_dist_dir:
        index_path = f"{resolved_settings.frontend_dist_dir}/index.html"
        if os.path.exists(index_path):
            assets_dir = os.path.join(resolved_settings.frontend_dist_dir, "assets")
            if os.path.isdir(assets_dir):
                try:
                    app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
                except Exception:
                    pass

    @app.get("/api/health")
    def health(request: Request) -> dict:
        return _service(request).health_snapshot()

    @app.get("/api/settings/runtime")
    def get_runtime_settings(request: Request) -> dict:
        return _service(request).get_runtime_settings()

    @app.post("/api/settings/runtime")
    def update_runtime_settings(payload: RuntimeSettingsUpdateRequest, request: Request) -> dict:
        return _service(request).update_runtime_settings(payload)

    @app.get("/api/jobs")
    def list_jobs(request: Request) -> list[dict]:
        return _service(request).list_job_snapshots()

    @app.post("/api/jobs/single")
    def create_single_job(payload: SingleJobRequest, request: Request) -> dict:
        return _service(request).create_single_job(payload)

    @app.post("/api/jobs/batch")
    def create_batch_job(payload: BatchJobRequest, request: Request) -> dict:
        return _service(request).create_batch_job(payload)

    @app.get("/api/wiki/health")
    def wiki_health(request: Request) -> dict:
        return _service(request).wiki_health()

    @app.post("/api/wiki/query")
    def wiki_query(payload: WikiQueryRequest, request: Request) -> dict:
        return _service(request).wiki_query(payload)

    @app.post("/api/wiki/reindex")
    def wiki_reindex(payload: WikiReindexRequest, request: Request) -> dict:
        return _service(request).create_wiki_reindex_job(payload)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict:
        detail = _service(request).get_job_detail(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="job_not_found")
        return detail

    @app.get("/api/jobs/{job_id}/state")
    def get_job_state(job_id: str, request: Request) -> dict:
        state = _service(request).get_job_state(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job_not_found")
        return state

    @app.get("/api/jobs/{job_id}/timeline")
    def get_job_timeline(job_id: str, request: Request) -> list[dict]:
        try:
            return _service(request).get_job_timeline(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.get("/api/jobs/{job_id}/logs")
    def get_job_logs(
        job_id: str,
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> dict:
        try:
            return _service(request).get_job_logs(job_id, offset=offset, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.get("/api/jobs/{job_id}/artifacts")
    def get_job_artifacts(job_id: str, request: Request) -> dict:
        try:
            return _service(request).get_job_artifacts(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.get("/api/jobs/{job_id}/download/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str, request: Request):
        try:
            path = _service(request).artifact_download_path(job_id, artifact_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact_not_found") from exc
        return FileResponse(path)

    @app.get("/api/jobs/{job_id}/artifact-json/{artifact_name}")
    def preview_artifact_json(job_id: str, artifact_name: str, request: Request):
        try:
            return _service(request).artifact_json_preview(job_id, artifact_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact_not_found_or_not_json") from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict:
        try:
            return _service(request).cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.post("/api/jobs/{job_id}/hitl/respond")
    def respond_hitl(job_id: str, payload: HitlResponseRequest, request: Request) -> dict:
        try:
            return _service(request).submit_hitl_response(job_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.post("/api/jobs/{job_id}/events/resume")
    def resume_event(job_id: str, payload: ExternalEventResumeRequest, request: Request) -> dict:
        try:
            return _service(request).resume_external_event(job_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.websocket("/ws/jobs")
    async def jobs_ws(websocket: WebSocket) -> None:
        await service.register_jobs_ws(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            service.unregister_ws(websocket)
        except Exception:
            service.unregister_ws(websocket)

    @app.websocket("/ws/jobs/{job_id}")
    async def job_ws(job_id: str, websocket: WebSocket) -> None:
        await service.register_detail_ws(job_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            service.unregister_ws(websocket)
        except Exception:
            service.unregister_ws(websocket)

    @app.get("/")
    def root() -> JSONResponse:
        index_path = f"{resolved_settings.frontend_dist_dir}/index.html" if resolved_settings.frontend_dist_dir else ""
        if index_path and os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse(
            {
                "name": "script_new Web Console",
                "status": "ok",
                "frontend": "/app/",
            }
        )

    @app.get("/app")
    @app.get("/app/")
    @app.get("/app/{path:path}")
    def spa_routes(path: str = ""):
        index_path = f"{resolved_settings.frontend_dist_dir}/index.html" if resolved_settings.frontend_dist_dir else ""
        if index_path and os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="frontend_not_built")

    return app
