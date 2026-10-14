import importlib
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from yoloview.engine import Engine, default_options
from yoloview.recording import Recorder


def test_stop_discards_inflight_result(tmp_path, monkeypatch):
    module = importlib.import_module('yoloview.engine')
    path = tmp_path / '中文图片.png'
    cv2.imencode('.png', np.zeros((40, 40, 3), np.uint8))[1].tofile(str(path))
    entered, release = threading.Event(), threading.Event()

    class BlockingRunner:
        options = default_options()
        names = {0: 'part'}

        def predict(self, frame):
            entered.set()
            assert release.wait(3)
            return frame, dict(count=1, counts={'part': 1}, inference_ms=1)

    monkeypatch.setattr(module, 'assets', SimpleNamespace(get=lambda *a: {'path': str(path), 'media_type': 'image'}))
    engine = Engine()
    engine.runner, engine.loaded_id = BlockingRunner(), 'model'
    engine.start(dict(kind='image', model_id='model', media_id='input', options=default_options()))
    assert entered.wait(3)
    with pytest.raises(ValueError, match='先停止'):
        engine.load('another-model', default_options())
    engine.stop_event.set()
    release.set()
    engine.thread.join(3)
    assert not engine.thread.is_alive()
    assert engine.status()['status'] == 'stopped'
    assert not engine.status()['has_frame']


def test_rejects_unknown_classes_before_start():
    engine = Engine()
    engine.runner = SimpleNamespace(options=default_options(), names={0: 'part'})
    engine.loaded_id = 'model'
    with pytest.raises(ValueError, match='类别编号'):
        engine.start(dict(kind='image', model_id='model', media_id='unused', options=default_options(classes=[10])))
    assert engine.thread is None


def test_recording_preserves_elapsed_time(tmp_path, monkeypatch):
    recording = importlib.import_module('yoloview.recording')
    clock = [100.0]
    monkeypatch.setattr(recording.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(recording, 'output_name', lambda *a: tmp_path / '标注视频.mp4')
    monkeypatch.setattr(recording, 'assets', SimpleNamespace(add=lambda path, *a, **k: {'path': str(path)}))
    # Only two inference results over four seconds must still produce a four-second video.
    recorder = Recorder(np.zeros((100, 100, 3), np.uint8))
    clock[0] = 102.0
    recorder.write(np.full((100, 100, 3), 255, np.uint8))
    clock[0] = 104.0
    output = recorder.finish()
    capture = cv2.VideoCapture(output['path'])
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    assert abs(duration - 4) <= 0.1
    assert int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little') in (b'avc1', b'h264')
    capture.release()
