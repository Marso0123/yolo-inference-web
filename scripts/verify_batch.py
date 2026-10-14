"""Actual model/API export acceptance. Run against an idle test service, default 8791."""
import csv
import hashlib
import inspect
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np
from ultralytics.data.utils import verify_image_label

ROOT = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url=sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8791', timeout=90)
report = []


def api(path, data=None):
    response = client.post('/api/' + path, json=data)
    response.raise_for_status()
    return response.json()


def wait():
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = client.get('/api/status').json()
        if state['status'] not in ('running', 'packaging', 'loading', 'stopping'):
            assert state['status'] != 'error', state
            return state
        time.sleep(.1)
    raise TimeoutError('任务未结束')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


items = json.loads((ROOT / 'data/assets.json').read_text('utf-8')).values()
bus = next(ROOT / i['path'] for i in items if i['name'] == 'bus.jpg' and i['kind'] == 'media')
inputs = ROOT / 'verification' / 'batch-inputs'
(inputs / '子目录').mkdir(parents=True, exist_ok=True)
(inputs / 'bus.jpg').write_bytes(bus.read_bytes())
(inputs / '子目录' / 'bus.jpg').write_bytes(bus.read_bytes())
cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1].tofile(inputs / '空白图.png')
(inputs / '坏图.jpg').write_bytes(b'broken image')
(inputs / '非图片.txt').write_text('folder chooser should ignore this file', 'utf-8')

print('validator:', inspect.signature(verify_image_label), flush=True)
for task in ('detect', 'segment'):
    for suffix in ('.pt', '.onnx'):
        for device in ('cpu', 'gpu'):
            name = 'yolo11n' + ('-seg' if task == 'segment' else '') + ('_640' if suffix == '.onnx' else '') + suffix
            model = next(i for i in items if i['name'] == name and i['kind'] == 'model')
            options = dict(device=device, imgsz=640, conf=.25, iou=.45, classes=[0], boxes=True, masks=True, labels=True)
            info = api('load', dict(model_id=model['id'], options=options))
            key = api('batches')['id']
            for index, source in enumerate((inputs / 'bus.jpg', inputs / '子目录' / 'bus.jpg', inputs / '空白图.png', inputs / '坏图.jpg')):
                data = dict(relative_path=source.relative_to(inputs).as_posix(), upload_id=str(index))
                response = client.post(f'/api/batches/{key}/upload', files={'file': (source.name, source.read_bytes())}, data=data)
                response.raise_for_status()
                if index == 0:
                    repeat = client.post(f'/api/batches/{key}/upload', files={'file': (source.name, source.read_bytes())}, data=data)
                    assert repeat.json()['total'] == 1
            remap = device == 'gpu'
            api('batch/start', dict(model_id=model['id'], options=options, batch_id=key, remap=remap, preview=device == 'gpu'))
            state = wait()
            job = state['batch']
            assert (job['done'], job['failed'], job['review']) == (3, 1, 1), job
            folder = Path(job['output_path'])
            rows = read_csv(folder / 'report.csv')
            assert len(rows) == 4
            names = (folder / 'classes.txt').read_text('utf-8').splitlines()
            assert len(names) == (1 if remap else 80)
            for row in rows:
                if row['status'] == '失败':
                    continue
                image, label = folder / row['image'], folder / row['label']
                assert image.stem == label.stem
                original = inputs / row['source']
                assert hashlib.sha256(image.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
                lines = label.read_text().splitlines()
                for line in lines:
                    values = line.split()
                    assert values[0] == '0'
                    assert len(values) == 5 if task == 'detect' else len(values) >= 7 and len(values) % 2 == 1
                    assert all(0 <= float(v) <= 1 for v in values[1:])
                # Same validator used by Ultralytics training data loading; no training is performed.
                checked = verify_image_label((str(image), str(label), '', False, len(names), 0, 0, False))
                assert checked[0] is not None and checked[8] == 0, checked
                if '空白' in row['source']:
                    assert label.stat().st_size == 0
                else:
                    assert len(lines) >= 1
            api(f'batches/{key}/archive')
            packed = wait()['batch']['archive']
            download = client.get('/api/download/' + packed['id'])
            download.raise_for_status()
            local_zip = Path(job['output_path']).with_suffix('.zip')
            assert download.content == local_zip.read_bytes()
            with zipfile.ZipFile(local_zip) as archive:
                assert archive.testzip() is None
                assert len([n for n in archive.namelist() if n.startswith('labels/')]) == 3
            report.append(dict(model=name, task=task, device=info['device'], cuda_nodes=info['cuda_nodes'],
                               job=job, training_label_validator='passed', download_bytes=len(download.content)))
            print(f'PASS {name} {device}: {job["done"]} saved, {job["review"]} review, ZIP verified', flush=True)
(ROOT / 'verification/batch-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
print('8 actual model export combinations passed', flush=True)
