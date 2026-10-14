"""Run after browser batch checks, against the idle QA server on 8791."""
import csv
import hashlib
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url='http://127.0.0.1:8791', timeout=60)


def post(path, data=None):
    response = client.post('/api/' + path, json=data)
    response.raise_for_status()
    return response.json()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


state = client.get('/api/status').json()
browser_job = state['batch']
assert (browser_job['done'], browser_job['failed'], browser_job['review']) == (2, 0, 1)
folder = Path(browser_job['output_path'])
assert (folder / 'classes.txt').read_text('utf-8').strip() == 'bus'
assert read_csv(folder / 'class_mapping.csv') == [{'original_id': '5', 'export_id': '0', 'name': 'bus'}]
assert sorted(p.read_text().strip().split(' ')[0] for p in (folder / 'labels').glob('*.txt')) == ['', '0']
folder_job = next(j for j in client.get('/api/batches').json()
                  if j['total'] == 4 and j.get('remap') and j.get('model', {}).get('classes') == [5] and j.get('archive'))
local_zip = Path(folder_job['output_path']).with_suffix('.zip')
download = ROOT / 'verification/browser-batch-downloaded.zip'
assert hashlib.sha256(local_zip.read_bytes()).digest() == hashlib.sha256(download.read_bytes()).digest()
result = {'browser_multi': browser_job, 'browser_folder_zip_sha256': hashlib.sha256(download.read_bytes()).hexdigest()}

model_id = state['model']['id']
options = dict(device='cpu', imgsz=640, conf=.25, iou=.45, classes=[5], boxes=True, masks=True, labels=True)
post('load', dict(model_id=model_id, options=options))
key = post('batches')['id']
bus = (ROOT / 'verification/batch-inputs/bus.jpg').read_bytes()
for index in range(30):
    response = client.post(f'/api/batches/{key}/upload', files={'file': ('bus.jpg', bus)}, data={'upload_id': str(index)})
    response.raise_for_status()
post('batch/start', dict(model_id=model_id, batch_id=key, options=options))
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    state = client.get('/api/status').json()
    if state['batch']['done'] >= 1:
        break
    time.sleep(.01)
else:
    raise TimeoutError('批量任务未产生结果')
post('stop')
cancelled = client.get('/api/status').json()['batch']
assert cancelled['status'] == 'cancelled' and 1 <= cancelled['done'] < 30, cancelled
folder = Path(cancelled['output_path'])
images = sorted(p.stem for p in (folder / 'images').glob('*'))
labels = sorted(p.stem for p in (folder / 'labels').glob('*'))
assert images == labels and len(labels) == cancelled['done']
assert len(read_csv(folder / 'report.csv')) == cancelled['done'] + cancelled['failed']
assert all(p.stat().st_size > 0 for p in (folder / 'labels').glob('*'))
result['actual_cancel'] = cancelled
# Restore stock demonstration model preferences changed solely for the browser check.
response = client.put(f'/api/models/{model_id}/class-filter', json={'classes': []})
response.raise_for_status()
(ROOT / 'verification/batch-edges-report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
print(f'PASS browser class 5 -> 0, empty label, downloaded ZIP SHA256, real cancel {cancelled["done"]}/30', flush=True)
