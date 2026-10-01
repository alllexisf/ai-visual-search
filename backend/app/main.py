import asyncio
import hmac
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .encoder import ClipEncoder
from .images import decode_image
from .reporting import hardware
from .retrieval import SearchIndex

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('visual-search')


class BodyLimit:
    """Bound actual bytes before multipart parsing, including chunked requests."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            total += len(chunk)
            if total > self.limit:
                return await JSONResponse({'detail': 'Request exceeds the upload size limit.'}, 413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get('more_body', False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


class RateLimiter:
    def __init__(self, settings):
        self.settings, self.buckets, self.last_prune = settings, {}, 0.0

    def check(self, ip, asset=False):
        now = time.monotonic()
        window = self.settings.rate_window
        if now - self.last_prune > window:
            self.buckets = {key: entries for key, entries in self.buckets.items() if entries and entries[-1] > now - window}
            self.last_prune = now
        key = (ip, asset)
        # Bound cardinality even if called with many distinct IPs.
        if key not in self.buckets and len(self.buckets) >= 10000:
            return window
        entries = self.buckets.setdefault(key, deque())
        while entries and entries[0] <= now - window:
            entries.popleft()
        limit = self.settings.asset_rate_limit if asset else self.settings.rate_limit
        if len(entries) >= limit:
            return max(1, int(entries[0] + window - now) + 1)
        entries.append(now)
        return None


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=12, ge=1, le=500)


class Result(BaseModel):
    id: int
    label: str
    dataset: str
    similarity: float
    thumbnail_url: str
    point: list[float]


class SearchResponse(BaseModel):
    results: list[Result]
    query_point: list[float]
    timings: dict[str, float]
    index_id: str


def create_app(settings=None, encoder_factory=ClipEncoder, index_factory=SearchIndex):
    cfg = settings or Settings()
    limiter = RateLimiter(cfg)

    @asynccontextmanager
    async def lifespan(app):
        cfg.validate()
        # Load exactly once per process. Run one Uvicorn worker in the provided packaging.
        app.state.index = await run_in_threadpool(index_factory, cfg.index_dir, cfg.model_id, cfg.model_revision)
        app.state.encoder = await run_in_threadpool(encoder_factory, cfg)
        app.state.model_slot = asyncio.Semaphore(1)
        app.state.hardware = hardware()
        yield

    app = FastAPI(title='AI Visual Search', version='1.0.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit, limit=cfg.max_upload_bytes + 64 * 1024)

    @app.middleware('http')
    async def security_and_timing(request: Request, call_next):
        start = time.perf_counter()
        request.state.embedding_ms = request.state.ranking_ms = 0.0
        response = None
        try:
            if request.url.path != '/health':
                provided = request.headers.get('X-API-Key', '')
                if not cfg.api_key or not hmac.compare_digest(provided.encode(), cfg.api_key.encode()):
                    response = JSONResponse({'detail': 'Invalid or missing API key.'}, 401)
                    return response
                retry = limiter.check(request.client.host if request.client else 'unknown', request.url.path.startswith('/thumbnails/'))
                if retry:
                    response = JSONResponse({'detail': 'Rate limit exceeded.'}, 429, headers={'Retry-After': str(retry)})
                    return response
            response = await call_next(request)
            response.headers['X-Request-Time-Ms'] = f'{(time.perf_counter() - start) * 1000:.2f}'
            return response
        finally:
            logger.info(json.dumps({'event': 'request', 'method': request.method, 'path': request.url.path,
                                    'status': response.status_code if response else 500,
                                    'embedding_ms': round(request.state.embedding_ms, 2),
                                    'ranking_ms': round(request.state.ranking_ms, 2),
                                    'total_ms': round((time.perf_counter() - start) * 1000, 2)}))

    # Outer CORS ensures errors also carry CORS headers. Preflight has no resource body
    # and is handled here; every real endpoint other than /health is authenticated.
    app.add_middleware(CORSMiddleware, allow_origins=[cfg.frontend_origin],
                       allow_methods=['GET', 'POST'], allow_headers=['X-API-Key', 'Content-Type'],
                       expose_headers=['Retry-After', 'X-Request-Time-Ms'])

    async def uploaded(file):
        try:
            data = await file.read(cfg.max_upload_bytes + 1)
            if not data:
                raise HTTPException(400, 'Choose a nonempty image.')
            if len(data) > cfg.max_upload_bytes:
                raise HTTPException(413, 'Image exceeds the upload size limit.')
            return await run_in_threadpool(decode_image, data, file.content_type, cfg.max_image_pixels)
        finally:
            await file.close()

    async def encode(request, function, value):
        start = time.perf_counter()
        async with app.state.model_slot:
            vector = await run_in_threadpool(function, value)
        request.state.embedding_ms = (time.perf_counter() - start) * 1000
        return vector

    async def ranked(request, vector, k, start):
        if k > cfg.max_top_k:
            raise HTTPException(422, f'top_k must be <= {cfg.max_top_k}.')
        tick = time.perf_counter()
        results = await run_in_threadpool(app.state.index.rank, vector, k)
        request.state.ranking_ms = (time.perf_counter() - tick) * 1000
        return {'results': results, 'query_point': app.state.index.project(vector),
                'index_id': app.state.index.manifest['index_id'],
                'timings': {'embedding_ms': request.state.embedding_ms, 'ranking_ms': request.state.ranking_ms,
                            'total_ms': (time.perf_counter() - start) * 1000}}

    @app.get('/health')
    async def health():
        index = app.state.index
        return {'status': 'ok', 'model': cfg.model_id, 'model_revision': cfg.model_revision,
                'dataset_size': len(index.metadata), 'classes': index.manifest['number_of_classes'],
                'index_id': index.manifest['index_id'], 'hardware': app.state.hardware,
                'torch_threads': cfg.torch_threads,
                'rate_limit': {'search_requests': cfg.rate_limit, 'window_seconds': cfg.rate_window}}

    @app.post('/search', response_model=SearchResponse)
    async def search(request: Request, file: UploadFile = File(...), top_k: int = Query(12, ge=1, le=cfg.max_top_k)):
        start = time.perf_counter()
        image = await uploaded(file)
        try:
            vector = (await encode(request, app.state.encoder.images, [image]))[0]
        finally:
            image.close()
        return await ranked(request, vector, top_k, start)

    @app.post('/search/text', response_model=SearchResponse)
    async def text_search(request: Request, body: TextQuery):
        start = time.perf_counter()
        query = body.query.strip()
        if not query:
            raise HTTPException(422, 'Enter a nonempty text query.')
        if body.top_k > cfg.max_top_k:
            raise HTTPException(422, f'top_k must be <= {cfg.max_top_k}.')
        vector = await encode(request, app.state.encoder.text, query)
        return await ranked(request, vector, body.top_k, start)

    @app.post('/embed')
    async def embed(request: Request, file: UploadFile = File(...)):
        image = await uploaded(file)
        try:
            vector = (await encode(request, app.state.encoder.images, [image]))[0]
        finally:
            image.close()
        return {'embedding': vector.tolist(), 'dimension': len(vector), 'model': cfg.model_id,
                'model_revision': cfg.model_revision, 'normalized': True}

    @app.get('/thumbnails/{image_id}')
    async def thumbnail(image_id: int):
        if image_id < 0 or image_id >= len(app.state.index.metadata):
            raise HTTPException(404, 'Thumbnail not found.')
        path = cfg.index_dir / 'thumbnails' / f'{image_id}.jpg'
        if not path.is_file():
            raise HTTPException(404, 'Thumbnail not found.')
        return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=86400', 'X-Content-Type-Options': 'nosniff'})

    return app


app = create_app()
