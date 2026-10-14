"""Real HTTP inference/export checks. Downloads official nano weights only for QA."""
import json
import time
import urllib.request
from pathlib import Path

import cv2
import httpx
import numpy as np

root = Path(__file__).resolve().parents[1]
qa = root / 'verification'
qa.mkdir(exist_ok=True)
client = httpx.Client(base_url='http://127.0.0.1:8790', timeout=180)


def call(route, data=None):
    response = client.post('/api/' + route, json=data)
    if response.status_code != 200:
        raise RuntimeError(f'{route}: {response.text}')
    return response.json()


def upload(path, kind):
    with path.open('rb') as source:
        response = client.post('/api/upload/' + kind, files={'file': (path.name, source)})
    response.raise_for_status()
    return response.json()


def wait(status, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get('/api/status').json()
        if state['status'] == 'error':
            raise RuntimeError(state['message'])
        if state['status'] == status:
            return state
        time.sleep(0.1)
    raise TimeoutError(state)


def infer(model, media, device, name):
    options = dict(device=device, imgsz=640, conf=0.25, iou=0.45, boxes=True, masks=True, labels=True)
    info = call('load', dict(model_id=model['id'], options=options))
    call('start', dict(model_id=model['id'], options=options, kind='image', media_id=media['id']))
    state = wait('finished')
    assert state['has_frame'] and state['count'] >= 1, state
    record = dict(name=name, device=device, task=info['task'], actual_device=info['device'],
                  cuda_nodes=info['cuda_nodes'], count=state['count'], counts=state['counts'],
                  inference_ms=state['inference_ms'])
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    bus = qa / 'bus.jpg'
    if not bus.exists():
        urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', bus)
    media = upload(bus, 'media')
    records = []
    exported = []
    for family, release in [('yolov8n', 'v8.3.0'), ('yolo11n', 'v8.3.0'), ('yolo26n', 'v8.4.0')]:
        for suffix in ['', '-seg']:
            name = family + suffix
            path = qa / (name + '.pt')
            if not path.exists():
                urllib.request.urlretrieve(f'https://github.com/ultralytics/assets/releases/download/{release}/{name}.pt', path)
            pt = upload(path, 'model')
            for device in ['cpu', 'gpu']:
                records.append(infer(pt, media, device, name + '.pt'))
            call('export', dict(model_id=pt['id'], imgsz=640))
            result = wait('ready')['export']
            assert result['status'] == 'done', result
            exported.append(dict(name=name, validation=result['validation'], id=result['result']['id']))
            for device in ['cpu', 'gpu']:
                records.append(infer(result['result'], media, device, name + '.onnx'))
            assert records[-4]['counts'] == records[-2]['counts'], (name, records[-4], records[-2])
    report = dict(hardware=client.get('/api/hardware').json(), inference=records, exports=exported,
                  sample='Ultralytics bus.jpg', field_accuracy_verified=False)
    (qa / 'models-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    # A reproducible video input using real imagery, without depending on a camera.
    frame = cv2.imdecode(np.fromfile(str(bus), dtype=np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (640, 480))
    video = cv2.VideoWriter(str(qa / 'sample-video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))
    for i in range(300):
        current = frame.copy()
        cv2.putText(current, f'{i / 25:.2f}s', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video.write(current)
    video.release()
    print('Completed:', len(records), 'real inferences and', len(exported), 'exports.', flush=True)
