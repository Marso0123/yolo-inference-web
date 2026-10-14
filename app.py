from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from yoloview.engine import engine
from yoloview.batch import IMAGE_TYPES, batches
from yoloview.inference import hardware
from yoloview.storage import MEDIA, MODELS, ROOT, assets, safe_name

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                    logging.FileHandler(ROOT / 'logs' / 'server.log', encoding='utf-8')])


@asynccontextmanager
async def lifespan(app):
    batches.recover()
    yield
    await asyncio.to_thread(engine.stop)
    if engine.runner:
        engine.runner.close()


app = FastAPI(title='YOLO 模型效果查看器', lifespan=lifespan)


def local_origin(origin):
    return not origin or urlsplit(origin).hostname in ('localhost', '127.0.0.1')


@app.middleware('http')
async def guard_origin(request, call_next):
    if not local_origin(request.headers.get('origin')):
        return JSONResponse({'detail': '只允许本机页面调用。'}, status_code=403)
    response = await call_next(request)
    if response.headers.get('content-type') == 'application/json':
        response.headers['content-type'] = 'application/json; charset=utf-8'
    return response


@app.exception_handler(ValueError)
async def value_error(request, error):
    return JSONResponse({'detail': str(error)}, status_code=400)


class Options(BaseModel):
    device: Literal['cpu', 'gpu'] = 'cpu'
    imgsz: int = Field(640, ge=32, le=2048, multiple_of=32)
    conf: float = Field(0.25, ge=0.01, le=1)
    iou: float = Field(0.45, ge=0.01, le=1)
    classes: list[StrictInt] | None = None
    boxes: bool = True
    masks: bool = True
    labels: bool = True


class Load(BaseModel):
    model_id: str
    options: Options = Field(default_factory=Options)


class Start(Load):
    kind: Literal['image', 'video', 'rtsp']
    media_id: str | None = None
    url: str = ''


class Export(BaseModel):
    model_id: str
    imgsz: int = Field(640, ge=32, le=2048, multiple_of=32)


class ClassFilter(BaseModel):
    classes: list[StrictInt] = Field(default_factory=list)


class BatchStart(Load):
    batch_id: str
    remap: bool = False
    preview: bool = False


@app.get('/api/hardware')
def get_hardware():
    return dict(hardware(), project_root=str(ROOT))


@app.get('/api/assets')
def get_assets():
    return assets.list()


@app.post('/api/upload/{kind}')
async def upload(kind: Literal['model', 'media'], file: UploadFile = File(...)):
    name = safe_name(file.filename or '')
    suffix = Path(name).suffix.lower()
    image_types = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_types = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.ts'}
    allowed = {'.pt', '.onnx'} if kind == 'model' else image_types | video_types
    if suffix not in allowed:
        raise ValueError('不支持的文件格式。')
    directory = (MODELS if kind == 'model' else MEDIA) / uuid.uuid4().hex
    directory.mkdir()
    path = directory / name
    total = 0
    try:
        with path.open('wb') as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 20 * 1024 ** 3:
                    raise ValueError('单个文件不能超过 20GB。')
                await asyncio.to_thread(target.write, chunk)
        if not total:
            raise ValueError('文件为空。')
        metadata = {'media_type': 'image' if suffix in image_types else 'video'} if kind == 'media' else {}
        return assets.add(path, name, kind, **metadata)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post('/api/load')
async def load_model(request: Load):
    return await asyncio.to_thread(engine.load, request.model_id, request.options.model_dump())


@app.put('/api/models/{model_id}/class-filter')
def save_class_filter(model_id: str, request: ClassFilter):
    return engine.save_class_preferences(model_id, request.classes)


@app.post('/api/start')
def start(request: Start):
    return engine.start(request.model_dump())


@app.post('/api/batches')
def create_batch():
    return batches.create()


@app.get('/api/batches')
def list_batches():
    return batches.list()


@app.post('/api/batches/{key}/upload')
async def upload_batch(key: str, file: UploadFile = File(...), relative_path: str = Form(''), upload_id: str = Form('')):
    name = safe_name(file.filename or '')
    if Path(name).suffix.lower() not in IMAGE_TYPES:
        raise ValueError('批量推理只接受 JPG／PNG／BMP／WebP 图片。')
    upload_id = upload_id or uuid.uuid4().hex
    if len(upload_id) > 64:
        raise ValueError('上传编号过长。')
    with batches.lock:
        job = batches.get(key)
        if job['status'] != 'uploading':
            raise ValueError('任务已开始，不能继续上传。')
        existing = any(i.get('upload_id') == upload_id for i in job['items'])
    if existing:
        await file.close()
        return batches.public(job)
    # UUID prevents concurrent uploads overwriting one another; user paths are metadata only.
    path = batches.folder(key) / 'inputs' / (uuid.uuid4().hex[:8] + '_' + name[-90:])
    total = 0
    try:
        with path.open('xb') as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 20 * 1024 ** 3:
                    raise ValueError('单张图片不能超过 20GB。')
                await asyncio.to_thread(target.write, chunk)
        if not total:
            raise ValueError('图片文件为空。')
        return batches.append(key, path, relative_path or name, upload_id)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post('/api/batch/start')
def start_batch(request: BatchStart):
    return engine.start_batch(request.model_dump())


@app.post('/api/batches/{key}/archive')
def archive_batch(key: str):
    return engine.archive_batch(key)


@app.post('/api/stop')
async def stop():
    return await asyncio.to_thread(engine.stop)


@app.get('/api/status')
def status():
    return engine.status()


@app.post('/api/save-image')
def save_image():
    return engine.save_image()


@app.post('/api/record/start')
def record_start():
    return engine.start_recording()


@app.post('/api/record/stop')
async def record_stop():
    item = await asyncio.to_thread(engine.finish_recording, True)
    if not item:
        raise ValueError('当前没有正在录制的视频。')
    return item


@app.post('/api/export')
def export(request: Export):
    return engine.export(request.model_id, request.imgsz)


@app.get('/api/download/{key}')
def download(key: str):
    item = assets.get(key)
    if item['kind'] not in ('output', 'model'):
        raise HTTPException(400, '文件不能下载。')
    return FileResponse(item['path'], filename=item['name'])


@app.websocket('/ws')
async def websocket(ws: WebSocket):
    if not local_origin(ws.headers.get('origin')):
        await ws.close(code=1008)
        return
    await ws.accept()
    previous = None
    disconnected = asyncio.create_task(ws.receive())
    try:
        while not disconnected.done():
            state, jpeg = engine.snapshot()
            identity = (state['generation'], state['seq'])
            if jpeg is not None and identity != previous:
                metadata = json.dumps(state, ensure_ascii=False).encode('utf-8')
                await ws.send_bytes(struct.pack('<I', len(metadata)) + metadata + jpeg)
                previous = identity
            await asyncio.sleep(0.025)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)


app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='static')
