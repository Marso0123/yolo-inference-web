"""Test video recording, real loopback RTSP, disconnection and reconnection."""
import json
import socket
import struct
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import httpx
from websockets.sync.client import connect

root = Path(__file__).resolve().parents[1]
qa = root / 'verification'
runtime = root / '.runtime' / 'qa-mediamtx'
runtime.mkdir(parents=True, exist_ok=True)
client = httpx.Client(base_url='http://127.0.0.1:8790', timeout=180)
CREATE = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def call(route, data=None):
    response = client.post('/api/' + route, json=data)
    response.raise_for_status()
    return response.json()


def wait(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get('/api/status').json()
        if state['status'] == 'error':
            raise RuntimeError(state['message'])
        if predicate(state):
            return state
        time.sleep(0.1)
    raise TimeoutError(state)


def recording_check(item, elapsed):
    response = client.get('/api/download/' + item['id'])
    response.raise_for_status()
    path = qa / ('saved-' + item['name'])
    path.write_bytes(response.content)
    capture = cv2.VideoCapture(str(path))
    fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    assert fps and count
    duration = count / fps
    assert abs(duration - elapsed) < 0.7, (duration, elapsed)
    ok, frame = capture.read()
    assert ok
    codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('ascii')
    capture.release()
    return dict(name=item['name'], duration=duration, elapsed=elapsed, codec=codec, frames=count)


def packet(ws):
    data = ws.recv(timeout=20)
    length = struct.unpack('<I', data[:4])[0]
    metadata = json.loads(data[4:4+length])
    assert data[4+length:4+length+2] == b'\xff\xd8'
    return metadata


if __name__ == '__main__':
    items = client.get('/api/assets').json()
    model = next(m for m in items if m['name'] == 'yolo11n_640.onnx')
    options = dict(device='gpu', imgsz=640)
    call('load', dict(model_id=model['id'], options=options))
    video = qa / 'sample-video.mp4'
    with video.open('rb') as file:
        response = client.post('/api/upload/media', files={'file': (video.name, file)})
    response.raise_for_status()
    media = response.json()
    call('start', dict(model_id=model['id'], options=options, kind='video', media_id=media['id']))
    wait(lambda s: s['has_frame'])
    with connect('ws://127.0.0.1:8790/ws', max_size=10 * 1024 * 1024) as ws:
        packets = [packet(ws) for _ in range(5)]
        assert all(a['generation'] == b['generation'] and a['seq'] < b['seq'] for a, b in zip(packets, packets[1:]))
    call('record/start')
    started = time.monotonic()
    time.sleep(2.5)
    recorded = call('record/stop')
    elapsed = time.monotonic() - started
    # Transcoding happens after recording is detached; use the observed frame timestamp.
    video_check = recording_check(recorded, 2.5)
    image = call('save-image')
    assert client.get('/api/download/' + image['id']).content.startswith(b'\xff\xd8')
    call('stop')

    executable = runtime / 'mediamtx.exe'
    if not executable.exists():
        release = json.load(urllib.request.urlopen('https://api.github.com/repos/bluenviron/mediamtx/releases/latest'))
        asset = next(a for a in release['assets'] if a['name'].endswith('windows_amd64.zip'))
        archive = runtime / 'mediamtx.zip'
        urllib.request.urlretrieve(asset['browser_download_url'], archive)
        with zipfile.ZipFile(archive) as zipped:
            with zipped.open('mediamtx.exe') as source, executable.open('wb') as target:
                target.write(source.read())
    config = runtime / 'qa.yml'
    config.write_text('logLevel: warn\nrtspEncryption: "no"\nrtspAddress: 127.0.0.1:18554\nrtspTransports: [tcp]\n'
                      'rtmp: no\nhls: no\nwebrtc: no\nsrt: no\napi: no\nmetrics: no\npprof: no\n'
                      'playback: no\npaths:\n  all_others:\n    source: publisher\n', 'utf-8')
    server = publisher = None
    log = (qa / 'rtsp-helper.log').open('w')
    publisher_log = (qa / 'rtsp-publisher.log').open('w')
    try:
        server = subprocess.Popen([str(executable), str(config)], stdout=log, stderr=log,
                                  creationflags=CREATE, cwd=runtime)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', 18554), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)

        def publish():
            return subprocess.Popen([str(root / '.runtime' / 'ffmpeg.exe'), '-hide_banner', '-loglevel', 'error',
                                     '-re', '-stream_loop', '-1', '-i', str(video), '-an', '-c:v', 'libx264',
                                     '-preset', 'ultrafast', '-tune', 'zerolatency', '-pix_fmt', 'yuv420p',
                                     '-f', 'rtsp', '-rtsp_transport', 'tcp', 'rtsp://127.0.0.1:18554/qa'],
                                    stdout=publisher_log, stderr=publisher_log, creationflags=CREATE)

        publisher = publish()
        time.sleep(1)
        call('start', dict(model_id=model['id'], options=options, kind='rtsp', url='rtsp://127.0.0.1:18554/qa'))
        connected = wait(lambda s: s['has_frame'] and s['status'] == 'running')
        call('record/start')
        time.sleep(1.5)
        publisher.terminate()
        publisher.wait(10)
        disconnected = wait(lambda s: s['status'] == 'reconnecting' and not s['has_frame'] and not s['saving'])
        assert not disconnected['recording']
        publisher = publish()
        reconnected = wait(lambda s: s['status'] == 'running' and s['has_frame'])
        assert reconnected['generation'] == connected['generation']
        assert reconnected['seq'] > connected['seq']
        call('stop')
        assert not client.get('/api/status').json()['recording']
        report = dict(video_recording=video_check, websocket_frames=len(packets), snapshot_download=True,
                      rtsp=dict(source='local H.264 RTSP via MediaMTX', connected=True, reconnect=True,
                                clears_stale_frame=True, closes_recording_on_disconnect=True),
                      real_camera_verified=False)
        (qa / 'streams-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        try:
            call('stop')
        except Exception:
            pass
        for process in [publisher, server]:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(10)
        log.close()
        publisher_log.close()
