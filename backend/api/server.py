"""ASGI entrypoint for the VN30 web app."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.features.assistant import POST_ROUTES as ASSISTANT_ROUTES
from api.features.market import GET_ROUTES, MarketState, load_snapshot
from api.features.profit_loss import POST_ROUTES as PROFIT_LOSS_ROUTES
from api.snapshot_cache import load_or_build_snapshot, read_snapshot

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
MAX_BODY_BYTES = 4096
LOGGER = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {"at": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                 "level": record.levelname, "logger": record.name,
                 "message": record.getMessage()}
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("api")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(worker: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        snapshot = read_snapshot() if worker else load_or_build_snapshot(load_snapshot)
        state = MarketState(snapshot)
        app.state.market = state
        stop = threading.Event()
        producer = None
        if not worker:
            producer = threading.Thread(target=state.run, args=(stop,), daemon=True)
            producer.start()
        try:
            yield
        finally:
            stop.set()
            if producer:
                producer.join(timeout=5)

    app = FastAPI(title="DSS VN30", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("request failed path=%s method=%s", request.url.path, request.method)
            raise
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        LOGGER.info("request path=%s method=%s status=%s duration_ms=%.1f",
                    request.url.path, request.method, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    def state_for(request: Request) -> MarketState:
        state = request.app.state.market
        if worker:
            state.refresh_from_cache()
        return state

    def get_route(route):
        def endpoint(request: Request):
            try:
                payload = route(state_for(request), parse_qs(request.url.query))
                response = JSONResponse(payload, headers={"Cache-Control": "private, no-cache"})
                etag = '"' + hashlib.sha256(response.body).hexdigest() + '"'
                if request.headers.get("if-none-match") == etag:
                    return Response(status_code=304, headers={"ETag": etag,
                                                              "Cache-Control": "private, no-cache"})
                response.headers["ETag"] = etag
                return response
            except ValueError as error:
                return JSONResponse({"error": str(error)}, status_code=400)
        return endpoint

    def post_route(route):
        async def endpoint(request: Request):
            content_length = request.headers.get("content-length", "0")
            if (not content_length.isdigit() or int(content_length) > MAX_BODY_BYTES
                    or request.headers.get("content-type", "").split(";")[0] != "application/json"):
                return JSONResponse({"error": "Yêu cầu không hợp lệ."}, status_code=400)
            body = await request.body()
            if not 0 < len(body) <= MAX_BODY_BYTES:
                return JSONResponse({"error": "Yêu cầu không hợp lệ."}, status_code=400)
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Yêu cầu phải là một đối tượng JSON.")
                return JSONResponse(route(state_for(request), payload))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                return JSONResponse({"error": str(error)}, status_code=400)
        return endpoint

    for path, route in GET_ROUTES.items():
        app.add_api_route(path, get_route(route), methods=["GET"])
    for path, route in (ASSISTANT_ROUTES | PROFIT_LOSS_ROUTES).items():
        app.add_api_route(path, post_route(route), methods=["POST"])

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(request: Request):
        try:
            state = state_for(request)
            market = state.market()
            if market.get("date") and market.get("stocks"):
                return {"status": "ok", "data_as_of": market["date"],
                        "quote_status": market["quote_status"]}
        except (FileNotFoundError, json.JSONDecodeError) as error:
            LOGGER.warning("Snapshot chưa sẵn sàng: %s", error)
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.get("/")
    def index():
        path = DIST / "index.html"
        if not path.exists():
            return JSONResponse({"error": "Chưa build giao diện."}, status_code=404)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")
    return app


worker_app = create_app(worker=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSS VN30 API")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--producer-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    configure_logging()
    if args.workers < 1:
        parser.error("--workers phải lớn hơn 0")
    if args.worker and (args.producer_only or args.build_only or args.rebuild):
        parser.error("--worker chỉ đọc snapshot đã tạo")
    if args.producer_only and args.build_only:
        parser.error("Chọn một chế độ tạo snapshot")
    if args.workers > 1 and not args.worker:
        parser.error("Nhiều worker cần một producer riêng")
    if args.build_only or args.rebuild:
        snapshot = load_or_build_snapshot(load_snapshot, args.rebuild)
        LOGGER.info("Snapshot sẵn sàng: %s", snapshot["date"])
        if args.build_only:
            return
    if args.producer_only:
        MarketState(load_or_build_snapshot(load_snapshot)).run(threading.Event())
        return
    application = "api.server:worker_app" if args.worker and args.workers > 1 else create_app(args.worker)
    uvicorn.run(application, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
