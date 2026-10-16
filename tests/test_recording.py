import importlib
import subprocess
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from yoloview.engine import Engine, default_options
from yoloview.recording import Recorder
from yoloview.storage import ROOT


def frame_with_id(index):
    frame = np.zeros((120, 200, 3), np.uint8)
    for bit in range(8):
        frame[8:24, 8 + bit * 20:24 + bit * 20] = 255 if index & (1 << bit) else 0
    frame[50:90, 10 + index:25 + index] = (0, 180, 240)
    return frame


def frame_id(frame):
    return sum((1 << bit) for bit in range(8)
               if frame[10:22, 10 + bit * 20:22 + bit * 20].mean() > 128)


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    ids = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        ids.append(frame_id(frame))
    capture.release()
    return fps, ids


@pytest.fixture
def recording_output(tmp_path, monkeypatch):
    module = importlib.import_module('yoloview.recording')
    path = tmp_path / '连续画面.mp4'
    monkeypatch.setattr(module, 'output_name', lambda *a: path)
    monkeypatch.setattr(module, 'assets', SimpleNamespace(add=lambda path, *a, **kw: dict(path=str(path), **kw)))
    return path


@pytest.mark.parametrize('fps', [25, 30, 50, 29.97])
def test_source_recording_keeps_frame_rate_and_every_frame(recording_output, fps):
    recorder = Recorder(frame_with_id(0), fps=fps, realtime=False, source_index=40)
    for index in range(1, 20):
        recorder.write(frame_with_id(index), source_index=40 + index)
    result = recorder.finish()
    actual_fps, ids = read_video(recording_output)
    assert abs(actual_fps - fps) < .01
    assert ids == list(range(20))
    assert result['frames'] == 20 and result['recording_mode'] == 'source_frames'


def test_source_gap_is_rejected_instead_of_silently_saved(recording_output):
    recorder = Recorder(frame_with_id(0), realtime=False, source_index=10)
    try:
        with pytest.raises(ValueError, match='不连续'):
            recorder.write(frame_with_id(2), source_index=12)
        assert not recorder.write(frame_with_id(0), source_index=10)
    finally:
        recorder.finish()


def test_realtime_writes_new_frame_at_its_timestamp(recording_output, monkeypatch):
    module = importlib.import_module('yoloview.recording')
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    recorder = Recorder(frame_with_id(0), fps=25)
    clock[0] = 100.04
    recorder.write(frame_with_id(1))
    clock[0] = 100.08
    recorder.write(frame_with_id(2))
    clock[0] = 100.12
    recorder.finish()
    fps, ids = read_video(recording_output)
    assert fps == 25 and ids == [0, 1, 2]


def test_realtime_fast_updates_do_not_lengthen_video(recording_output, monkeypatch):
    module = importlib.import_module('yoloview.recording')
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    recorder = Recorder(frame_with_id(0), fps=25)
    for index in range(1, 40):
        clock[0] = 100 + index / 100
        recorder.write(frame_with_id(index))
    clock[0] = 100.4
    recorder.finish()
    fps, ids = read_video(recording_output)
    assert len(ids) == 10
    assert len(ids) / fps == .4


def test_writer_backpressure_does_not_lock_engine_status(recording_output, monkeypatch):
    engine = Engine()
    engine.state['source_kind'] = 'rtsp'
    entered, release = threading.Event(), threading.Event()
    recorder = Recorder(frame_with_id(0))
    original = recorder.write

    def blocking_write(*args, **kw):
        entered.set()
        assert release.wait(3)
        return original(*args, **kw)

    monkeypatch.setattr(recorder, 'write', blocking_write)
    engine.recorder = recorder
    worker = threading.Thread(target=engine.publish,
                              args=(frame_with_id(1), dict(count=1, fps=25, inference_ms=1)))
    worker.start()
    assert entered.wait(3)
    acquired = engine.lock.acquire(timeout=.2)
    if acquired:
        engine.lock.release()
    release.set()
    worker.join(3)
    engine.finish_recording()
    assert acquired, 'disk backpressure held the Engine state lock'


@pytest.mark.parametrize('source_codec', ['mjpeg', 'h264'])
def test_slow_inference_and_inflight_record_start_keep_all_source_frames(tmp_path, recording_output, monkeypatch, source_codec):
    module = importlib.import_module('yoloview.engine')
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (200, 120))
    for index in range(16):
        writer.write(frame_with_id(index))
    writer.release()
    if source_codec == 'h264':
        encoded = tmp_path / 'source.mp4'
        completed = subprocess.run([str(ROOT / '.runtime/ffmpeg.exe'), '-hide_banner', '-loglevel', 'error',
                                    '-y', '-i', str(source), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(encoded)],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        assert completed.returncode == 0
        source = encoded
    entered, release = threading.Event(), threading.Event()

    class SlowRunner:
        names = {0: 'object'}
        options = default_options()
        calls = 0

        def predict(self, frame):
            self.calls += 1
            if self.calls == 4:
                entered.set()
                assert release.wait(4)
            time.sleep(.08)  # Deliberately much slower than the 30 FPS source.
            return frame, dict(count=1, counts={'object': 1}, inference_ms=80)

    monkeypatch.setattr(module, 'assets', SimpleNamespace(get=lambda *a: {'path': str(source), 'media_type': 'video'}))
    engine = Engine()
    engine.runner, engine.loaded_id = SlowRunner(), 'model'
    engine.start(dict(model_id='model', media_id='source', kind='video', options=default_options()))
    assert entered.wait(4)
    start_index = engine.frame_index
    assert start_index > 0
    started = engine.start_recording()
    assert started['recording_fps'] == 30
    release.set()
    engine.thread.join(10)
    assert not engine.thread.is_alive() and engine.status()['status'] == 'finished'
    fps, ids = read_video(recording_output)
    assert fps == 30
    assert ids == list(range(start_index, 16))  # Even the frames the earlier preview would skip.
    assert engine.status()['last_saved']['frames'] == 16 - start_index


@pytest.fixture
def short_video_engine(tmp_path, monkeypatch):
    source = tmp_path / '一秒短视频.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (200, 120))
    for index in range(25):
        writer.write(frame_with_id(index))
    writer.release()
    module = importlib.import_module('yoloview.engine')
    monkeypatch.setattr(module, 'assets', SimpleNamespace(get=lambda *a: {'path': str(source), 'media_type': 'video'}))

    class Runner:
        options = default_options()

        def predict(self, frame):
            time.sleep(.055)  # Slower than source FPS: full recording must never skip.
            return frame, dict(count=1, counts={'object': 1}, inference_ms=55)

    engine = Engine()
    engine.runner, engine.loaded_id = Runner(), 'model'
    return engine


def start_full(engine, **changes):
    return engine.start(dict(model_id='model', kind='video', media_id='clip',
                             options=default_options(), save_full_video=True, **changes))


@pytest.mark.parametrize('count', [1, 25])
def test_full_video_records_first_through_last_without_second_click(short_video_engine, recording_output, count):
    engine = short_video_engine
    if count == 1:
        # Replace the input with an extremely short one-frame video.
        module = importlib.import_module('yoloview.engine')
        source = module.assets.get('clip')['path']
        writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'MJPG'), 25, (200, 120))
        writer.write(frame_with_id(0))
        writer.release()
    start_full(engine)
    engine.thread.join(10)
    state = engine.status()
    assert state['status'] == 'finished' and not state['saving'] and not state['recording']
    assert not state['record_armed'] and not state['record_error']
    assert '已保存' in state['message']
    fps, ids = read_video(recording_output)
    assert fps == 25 and ids == list(range(count))
    assert state['recording_result'] == state['last_saved']
    assert state['last_saved']['duration'] == count / 25


def test_default_preview_does_not_record(short_video_engine, recording_output):
    engine = short_video_engine
    engine.start(dict(model_id='model', kind='video', media_id='clip', options=default_options()))
    engine.thread.join(10)
    assert engine.status()['status'] == 'finished'
    assert engine.status()['recording_result'] is None and not recording_output.exists()


@pytest.mark.parametrize('first', ['eof', 'manual'])
def test_eof_and_manual_stop_share_one_result(short_video_engine, recording_output, monkeypatch, first):
    engine = short_video_engine
    entered, release = threading.Event(), threading.Event()
    original = Recorder.finish
    calls = []

    def finish(recorder):
        calls.append(1)
        entered.set()
        assert release.wait(8)
        return original(recorder)

    monkeypatch.setattr(Recorder, 'finish', finish)
    start_full(engine)
    manual_results, manual_errors = [], []

    def manual():
        try:
            manual_results.append(engine.finish_recording(True))
        except Exception as error:
            manual_errors.append(str(error))

    if first == 'manual':
        deadline = time.monotonic() + 4
        while not engine.status()['recording'] and time.monotonic() < deadline:
            time.sleep(.01)
        worker = threading.Thread(target=manual)
        worker.start()
    assert entered.wait(6)
    if first == 'eof':
        assert engine.status()['status'] == 'finalizing'
        with pytest.raises(ValueError):
            engine.ensure_idle()
        worker = threading.Thread(target=manual)
        worker.start()
    # Even when manual save detached the recorder, EOF must wait for it.
    deadline = time.monotonic() + 4
    while engine.status()['status'] != 'finalizing' and time.monotonic() < deadline:
        time.sleep(.01)
    assert engine.status()['status'] == 'finalizing'
    assert engine.status()['saving'] and not engine.status()['recording_result']
    release.set()
    worker.join(10)
    engine.thread.join(10)
    assert not worker.is_alive() and not engine.thread.is_alive()
    assert not manual_errors and len(calls) == 1
    assert manual_results == [engine.status()['recording_result']]
    assert engine.finish_recording(True) == manual_results[0]  # Repeated stop is idempotent.
    assert engine.status()['status'] == 'finished' and not engine.status()['saving']


def test_stopping_full_video_saves_processed_prefix(short_video_engine, recording_output):
    engine = short_video_engine
    start_full(engine)
    deadline = time.monotonic() + 4
    while (engine.frame_index or 0) < 3 and time.monotonic() < deadline:
        time.sleep(.01)
    state = engine.stop()
    fps, ids = read_video(recording_output)
    assert state['status'] == 'stopped' and '已保存' in state['message']
    assert fps == 25 and 1 <= len(ids) < 25 and ids == list(range(len(ids)))


def test_encode_failure_stays_visible_and_keeps_avi(short_video_engine, recording_output, monkeypatch):
    module = importlib.import_module('yoloview.recording')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stderr=b'test encoder failure'))
    engine = short_video_engine
    start_full(engine)
    engine.thread.join(10)
    state = engine.status()
    assert '保存失败' in state['message'] and 'AVI' in state['record_error']
    assert state['recording_result'] is None and not state['saving']
    assert recording_output.with_suffix('.avi').exists()
    with pytest.raises(ValueError, match='AVI'):
        engine.finish_recording(True)


def test_full_recording_rejected_for_rtsp(short_video_engine):
    with pytest.raises(ValueError, match='仅支持本地视频'):
        short_video_engine.start(dict(model_id='model', kind='rtsp', url='rtsp://localhost/test',
                                     options=default_options(), save_full_video=True))


def test_inference_error_preserved_when_partial_recording_is_saved(short_video_engine, recording_output):
    engine = short_video_engine
    predict = engine.runner.predict
    calls = []

    def fail(frame):
        calls.append(1)
        if len(calls) == 4:
            raise ValueError('测试推理异常')
        return predict(frame)

    engine.runner.predict = fail
    start_full(engine)
    engine.thread.join(10)
    state = engine.status()
    assert state['status'] == 'error'
    assert '测试推理异常' in state['message'] and '已保存' in state['message']
    assert read_video(recording_output)[1] == [0, 1, 2]
