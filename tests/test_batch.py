import csv
import importlib
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.engine.results import Results

from yoloview.batch import BatchStore, label_rows, run_dataset, archive_dataset
from yoloview.engine import Engine, default_options


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


@pytest.fixture
def setup_batch(tmp_path, monkeypatch):
    module = importlib.import_module('yoloview.batch')
    root = tmp_path / 'datasets'
    root.mkdir()
    monkeypatch.setattr(module, 'DATASETS', root)
    store = BatchStore(tmp_path / 'inbox')
    key = store.create()['id']

    def add(name, data=None):
        path = store.folder(key) / 'inputs' / f'{len(store.get(key)["items"])}_{Path(name).name}'
        path.write_bytes(data if data is not None else cv2.imencode('.png', np.full((40, 60, 3), 100, np.uint8))[1].tobytes())
        store.append(key, path, name, str(len(store.get(key)['items'])))
        return path

    return store, key, add


class FixtureRunner:
    names = {0: '背景物体', 1: 'person', 2: 'part'}

    def __init__(self, task='detect', empty=False):
        self.model = SimpleNamespace(task=task)
        self.options = default_options(classes=[2])
        self.empty = empty

    def predict_result(self, frame):
        boxes = np.empty((0, 6), np.float32) if self.empty else np.array([[6, 4, 30, 20, .9, 2]], np.float32)
        mask = None
        if self.model.task == 'segment':
            mask = np.zeros((len(boxes), 40, 60), np.uint8)
            if len(boxes):
                mask[0, 4:20, 6:30] = 1
        return Results(frame, 'image', self.names, boxes=torch.from_numpy(boxes),
                       masks=torch.from_numpy(mask) if mask is not None else None), 5

    def render_result(self, result, elapsed):
        return result.orig_img.copy(), dict(count=len(result.boxes), counts={}, inference_ms=elapsed)

    def close(self):
        pass


@pytest.mark.parametrize('task', ['detect', 'segment'])
@pytest.mark.parametrize('remap', [False, True])
def test_yolo_labels_and_original_pairs(setup_batch, task, remap):
    store, key, add = setup_batch
    first = add('子目录/同名.png')
    second = add('另一个目录/同名.png')
    runner = FixtureRunner(task)
    job = run_dataset(store, key, runner, threading.Event(), lambda *a: None, lambda *a: None,
                      {'name': 'best.pt'}, remap=remap, preview=True)
    folder = Path(job['output_path'])
    assert job['done'] == 2 and job['failed'] == 0 and job['status'] == 'completed'
    images = sorted((folder / 'images').glob('*'))
    assert images[0].read_bytes() == first.read_bytes()
    assert images[1].read_bytes() == second.read_bytes()
    assert images[0].stem != images[1].stem
    for image in images:
        line = (folder / 'labels' / (image.stem + '.txt')).read_text().split()
        assert int(line[0]) == (0 if remap else 2)
        assert all(0 <= float(c) <= 1 for c in line[1:])
        assert len(line) == 5 if task == 'detect' else len(line) >= 7 and len(line) % 2 == 1
    assert len(list((folder / 'preview').glob('*.jpg'))) == 2
    assert (folder / 'classes.txt').read_text('utf-8').splitlines() == (['part'] if remap else list(runner.names.values()))
    assert len(rows(folder / 'report.csv')) == 2
    assert rows(folder / 'review.csv') == []


def test_empty_labels_and_bad_image(setup_batch):
    store, key, add = setup_batch
    add('未检出.png')
    add('坏图.jpg', b'not an image')
    job = run_dataset(store, key, FixtureRunner(empty=True), threading.Event(), lambda *a: None,
                      lambda *a: None, {'name': 'best.pt'})
    folder = Path(job['output_path'])
    assert (job['done'], job['failed'], job['review']) == (1, 1, 1)
    assert job['status'] == 'completed_with_errors'
    assert next((folder / 'labels').glob('*.txt')).read_bytes() == b''
    assert len(list((folder / 'images').glob('*'))) == 1
    assert len(rows(folder / 'review.csv')) == 1
    assert [r['status'] for r in rows(folder / 'report.csv')] == ['待复核', '失败']


def test_cancel_preserves_completed_pairs(setup_batch):
    store, key, add = setup_batch
    add('one.png')
    add('two.png')
    stop = threading.Event()
    job = run_dataset(store, key, FixtureRunner(), stop, lambda *a: stop.set(), lambda *a: None, {'name': 'best.pt'})
    folder = Path(job['output_path'])
    assert job['status'] == 'cancelled' and job['done'] == 1
    assert len(list((folder / 'images').glob('*'))) == len(list((folder / 'labels').glob('*'))) == 1
    assert len(rows(folder / 'report.csv')) == 1


def test_upload_retry_and_restart_recovery(setup_batch):
    store, key, add = setup_batch
    path = add('a.png')
    duplicate = path.with_name('duplicate.png')
    duplicate.write_bytes(path.read_bytes())
    assert store.append(key, duplicate, 'a.png', '0')['total'] == 1
    assert not duplicate.exists()
    store.update(key, status='running')
    store.recover()
    assert store.get(key)['status'] == 'interrupted'
    with pytest.raises(ValueError):
        store.folder('../../escape')


def test_exif_orientation_matches_saved_pixels(setup_batch):
    store, key, add = setup_batch
    image = Image.new('RGB', (60, 40), 'red')
    exif = image.getexif()
    exif[274] = 6
    path = store.folder(key) / 'inputs' / 'oriented.jpg'
    image.save(path, exif=exif)
    store.append(key, path, 'oriented.jpg', '0')
    job = run_dataset(store, key, FixtureRunner(empty=True), threading.Event(), lambda *a: None,
                      lambda *a: None, {'name': 'best.pt'})
    output = next((Path(job['output_path']) / 'images').glob('*'))
    assert output.suffix == '.png'
    assert cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (60, 40)


def test_degenerate_polygon_is_reviewed():
    result = FixtureRunner('segment').predict_result(np.zeros((40, 60, 3), np.uint8))[0]
    result.masks.data[:] = 0
    assert label_rows(result, 'segment', {2: 0}) == ([], 1)


def test_zip_contains_dataset_and_cancel_removes_temporary(setup_batch, monkeypatch):
    store, key, add = setup_batch
    add('a.png')
    job = run_dataset(store, key, FixtureRunner(), threading.Event(), lambda *a: None, lambda *a: None, {'name': 'best.pt'})
    module = importlib.import_module('yoloview.batch')
    monkeypatch.setattr(module, 'assets', SimpleNamespace(add=lambda path, *a, **k: {'path': str(path)}))
    item = archive_dataset(store, key, threading.Event())
    with zipfile.ZipFile(item['path']) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.startswith('images/')]) == 1
        assert 'review.csv' in archive.namelist()
    stop = threading.Event()
    stop.set()
    with pytest.raises(ValueError, match='取消'):
        archive_dataset(store, key, stop)
    assert not Path(job['output_path']).with_suffix('.zip.tmp').exists()


def test_batch_excludes_simultaneous_inference(setup_batch, monkeypatch):
    store, key, add = setup_batch
    add('a.png')
    module = importlib.import_module('yoloview.engine')
    monkeypatch.setattr(module, 'batches', store)
    entered, release = threading.Event(), threading.Event()

    class BlockingRunner(FixtureRunner):
        def predict_result(self, frame):
            entered.set()
            assert release.wait(5)
            return super().predict_result(frame)

    engine = Engine()
    engine.runner, engine.loaded_id = BlockingRunner(), 'model'
    engine.state['model'] = {'name': 'best.pt'}
    engine.start_batch(dict(model_id='model', batch_id=key, options=default_options(classes=[2]), remap=False, preview=False))
    assert entered.wait(5)
    with pytest.raises(ValueError, match='先停止'):
        engine.start(dict(model_id='model'))
    engine.stop_event.set()
    release.set()
    engine.thread.join(5)
    assert engine.status()['batch']['status'] == 'cancelled'


def test_batch_api_validation_and_upload_idempotency(setup_batch, monkeypatch):
    from fastapi.testclient import TestClient
    app_module = importlib.import_module('app')
    engine_module = importlib.import_module('yoloview.engine')
    store, _, _ = setup_batch
    engine = Engine()
    engine.runner, engine.loaded_id = FixtureRunner(), 'model'
    engine.state['model'] = {'name': 'best.pt'}
    monkeypatch.setattr(app_module, 'engine', engine)
    monkeypatch.setattr(app_module, 'batches', store)
    monkeypatch.setattr(engine_module, 'batches', store)
    image = cv2.imencode('.png', np.zeros((40, 60, 3), np.uint8))[1].tobytes()
    with TestClient(app_module.app) as client:
        key = client.post('/api/batches').json()['id']
        for _ in range(2):
            response = client.post(f'/api/batches/{key}/upload', files={'file': ('中文.png', image)},
                                   data={'relative_path': '../outside/中文.png', 'upload_id': '0'})
            assert response.status_code == 200 and response.json()['total'] == 1
        assert len(list((store.folder(key) / 'inputs').glob('*'))) == 1
        assert client.post(f'/api/batches/{key}/upload', files={'file': ('bad.exe', image)}).status_code == 400
        request = dict(model_id='model', batch_id=key, options=default_options(classes=[True]))
        assert client.post('/api/batch/start', json=request).status_code == 422
        request['options']['classes'] = [99]
        assert client.post('/api/batch/start', json=request).status_code == 400
        request['options']['classes'] = [2]
        assert client.post('/api/batch/start', json=request).status_code == 200
        engine.thread.join(5)
        assert engine.status()['batch']['done'] == 1
        assert client.post('/api/batch/start', json=request).status_code == 400
        assert client.post(f'/api/batches/{key}/upload', files={'file': ('late.png', image)}).status_code == 400
