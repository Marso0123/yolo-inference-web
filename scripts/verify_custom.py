"""Verify the user-trained helmet checkpoint and its web-exported ONNX."""
import json
import time
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url='http://127.0.0.1:8790', timeout=120)
items = client.get('/api/assets').json()
models = [next(i for i in items if i['name'] == name) for name in ['best.pt', 'best_640.onnx']]
media = next(i for i in items if i['name'] == 'shel5k_hard_hat_workers1009.png')
results = []
for model in models:
    for device in ['cpu', 'gpu']:
        options = dict(device=device, imgsz=640)
        response = client.post('/api/load', json=dict(model_id=model['id'], options=options))
        response.raise_for_status()
        info = response.json()
        response = client.post('/api/start', json=dict(model_id=model['id'], options=options,
                                                       media_id=media['id'], kind='image'))
        response.raise_for_status()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = client.get('/api/status').json()
            if state['status'] == 'error':
                raise RuntimeError(state['message'])
            if state['status'] == 'finished':
                break
            time.sleep(0.05)
        assert state['has_frame'] and state['counts'] == {'helmet': 4}, state
        results.append(dict(name=model['name'], device=device, info=info, count=state['count'],
                            counts=state['counts'], inference_ms=state['inference_ms']))
report = dict(model='helmet_head_yolo11n_640_v1/weights/best.pt', input=media['name'],
              test='真实图片推理与格式兼容性', results=results, accuracy_evaluation=False,
              browser_upload=True, browser_export=True, browser_snapshot=True)
(root / 'verification' / 'custom-model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
print(json.dumps(report, ensure_ascii=False), flush=True)
