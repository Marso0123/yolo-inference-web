from __future__ import annotations

import csv
import json
import shutil
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .storage import DATA, assets, safe_name

IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
INBOX = DATA / 'batches'
DATASETS = DATA / 'datasets'
for directory in (INBOX, DATASETS):
    directory.mkdir(exist_ok=True)


class BatchStore:
    def __init__(self, root=INBOX):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def folder(self, key):
        if len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('批量任务不存在。')
        return self.root / key

    def get(self, key):
        with self.lock:
            path = self.folder(key) / 'job.json'
            if not path.is_file():
                raise ValueError('批量任务不存在。')
            return json.loads(path.read_text('utf-8'))

    def update(self, key, **changes):
        with self.lock:
            job = self.get(key)
            job.update(changes)
            self.write(job)
            return self.public(job)

    def write(self, job):
        path = self.folder(job['id']) / 'job.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), 'utf-8')
        temp.replace(path)

    def create(self):
        with self.lock:
            key = uuid.uuid4().hex
            (self.folder(key) / 'inputs').mkdir(parents=True)
            job = dict(id=key, created=datetime.now().isoformat(timespec='seconds'), status='uploading',
                       items=[], total=0, done=0, failed=0, review=0, current='', output_path=None,
                       archive=None, message='等待上传图片')
            self.write(job)
            return self.public(job)

    @staticmethod
    def public(job):
        return {k: v for k, v in job.items() if k != 'items'}

    def list(self):
        with self.lock:
            jobs = []
            for path in sorted(self.root.glob('*/job.json'), reverse=True):
                job = json.loads(path.read_text('utf-8'))
                jobs.append(self.public(job))
            return sorted(jobs, key=lambda j: j['created'], reverse=True)

    def recover(self):
        """Called once by the server lifespan, before it accepts jobs."""
        with self.lock:
            for path in self.root.glob('*/job.json'):
                job = json.loads(path.read_text('utf-8'))
                if job['status'] in ('running', 'packaging'):
                    job.update(status='interrupted', message='上次服务中断；已保存结果保留，请复核 report.csv。')
                    self.write(job)

    def append(self, key, path, original, upload_id):
        with self.lock:
            job = self.get(key)
            if job['status'] != 'uploading':
                raise ValueError('该任务已开始，不能继续上传。')
            if any(i.get('upload_id') == upload_id for i in job['items']):
                path.unlink(missing_ok=True)
                return self.public(job)
            job['items'].append(dict(path=path.name, original=original[:1024], upload_id=upload_id))
            job['total'] = len(job['items'])
            self.write(job)
            return self.public(job)


def label_rows(result, task, mapping):
    """Standard training labels: no confidence or tracking columns."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], 0
    classes = boxes.cls.cpu().numpy().astype(int)
    if task == 'segment':
        if result.masks is None or len(result.masks) != len(boxes):
            raise ValueError('分割框与掩膜数量不一致，不能导出分割标签。')
        coordinates = result.masks.xyn
    else:
        coordinates = boxes.xywhn.cpu().numpy()
    rows, skipped = [], 0
    for class_id, points in zip(classes, coordinates):
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            raise ValueError('标签坐标包含无效数值。')
        points = np.clip(points, 0, 1)
        if task == 'segment' and (len(points) < 3 or abs(cv2.contourArea(points.astype(np.float32))) <= 1e-12):
            skipped += 1
            continue
        if task == 'detect' and (points[2] <= 0 or points[3] <= 0):
            skipped += 1
            continue
        rows.append(' '.join([str(mapping[int(class_id)])] + [f'{v:.8f}' for v in points.reshape(-1)]))
    return rows, skipped


def run_dataset(store, key, runner, stop, publish, progress, model, remap=False, preview=False):
    job = store.get(key)
    folder = DATASETS / f'dataset_{datetime.now():%Y%m%d_%H%M%S}_{key[:8]}'
    folder.mkdir()
    for name in ('images', 'labels'):
        (folder / name).mkdir()
    if preview:
        (folder / 'preview').mkdir()
    selected = runner.options.get('classes') or sorted(runner.names)
    ids = sorted(set(selected)) if remap else sorted(runner.names)
    mapping = {old: new if remap else old for new, old in enumerate(ids)}
    # Standard Ultralytics models have a continuous original class dictionary.
    if not remap and ids != list(range(len(ids))):
        raise ValueError('模型类别编号不连续，请启用连续编号。')
    (folder / 'classes.txt').write_text('\n'.join(runner.names[i] for i in ids) + '\n', 'utf-8')
    with (folder / 'class_mapping.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['original_id', 'export_id', 'name'])
        writer.writerows((i, mapping[i], runner.names[i]) for i in ids)
    (folder / '说明.txt').write_text(
        f"模型：{model['name']}\n任务：{runner.model.task}\n参数：{json.dumps(runner.options, ensure_ascii=False)}\n"
        'images 为未叠加标注的图像；带 EXIF 方向的图片转为正向 PNG，其余保留原文件字节。\n'
        '标签为自动预标注，请复核误检、漏检、分割轮廓后用于训练。\n'
        '空标签和退化目标列入 review.csv，不代表已确认的负样本。\n'
        '未划分训练/验证集。classes.txt 行号即导出类别编号。\n'
        '取消、失败或服务中断的任务只包含已保存的完整图片/标签对；以 report.csv 为准。\n', 'utf-8')
    counters = dict(done=0, failed=0, review=0)
    store.update(key, output_path=str(folder), **counters)
    headers = ['source', 'image', 'label', 'status', 'objects', 'inference_ms', 'note']
    with (folder / 'report.csv').open('w', newline='', encoding='utf-8-sig') as report, \
            (folder / 'review.csv').open('w', newline='', encoding='utf-8-sig') as review:
        writer, reviewer = csv.writer(report), csv.writer(review)
        writer.writerow(headers)
        reviewer.writerow(headers)
        report.flush()
        review.flush()
        for index, item in enumerate(job['items'], 1):
            if stop.is_set():
                break
            source = store.folder(key) / 'inputs' / item['path']
            stem = f'{index:06d}_{Path(safe_name(item["original"])).stem[:80]}'
            image_path = folder / 'images' / (stem + source.suffix.lower())
            label_path = folder / 'labels' / (stem + '.txt')
            preview_path = folder / 'preview' / (stem + '.jpg')
            progress(store.update(key, current=item['original'], **counters))
            try:
                frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('图片无法解码')
                result, elapsed = runner.predict_result(frame)
                if stop.is_set():
                    break
                rows, skipped = label_rows(result, runner.model.task, mapping)
                note = f'{skipped} 个退化目标未写入标签；需复核' if skipped else ''
                with Image.open(source) as image:
                    orientation = image.getexif().get(274, 1)
                if orientation not in (None, 1):
                    image_path = image_path.with_suffix('.png')
                    ok, encoded = cv2.imencode('.png', frame)
                    if not ok:
                        raise ValueError('图片方向规范化失败')
                    encoded.tofile(image_path)
                    note += '；已按 EXIF 方向规范化为 PNG'
                else:
                    shutil.copyfile(source, image_path)
                label_path.write_text('\n'.join(rows) + ('\n' if rows else ''), 'utf-8')
                plotted, stats = runner.render_result(result, elapsed)
                if preview:
                    ok, encoded = cv2.imencode('.jpg', plotted)
                    if not ok:
                        raise ValueError('预览图编码失败')
                    encoded.tofile(preview_path)
                needs_review = not rows or skipped > 0
                if not rows:
                    note = '未检出有效目标，空标签待复核。' + note
                row = [item['original'], image_path.relative_to(folder).as_posix(),
                       label_path.relative_to(folder).as_posix(), '待复核' if needs_review else '已保存',
                       len(rows), round(elapsed, 1), note]
                publish(plotted, dict(stats, fps=0))
                if needs_review:
                    reviewer.writerow(row)
                    review.flush()
                    counters['review'] += 1
                counters['done'] += 1
            except Exception as error:
                image_path.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
                preview_path.unlink(missing_ok=True)
                counters['failed'] += 1
                row = [item['original'], '', '', '失败', 0, '', str(error)]
            writer.writerow(row)
            report.flush()
            progress(store.update(key, **counters))
    status = 'cancelled' if stop.is_set() else ('completed_with_errors' if counters['failed'] else 'completed')
    message = '已取消，保留已完成结果' if stop.is_set() else '批量推理完成'
    return store.update(key, status=status, current='', message=message, **counters)


def archive_dataset(store, key, stop):
    job = store.get(key)
    if not job.get('output_path'):
        raise ValueError('该任务尚未生成数据集。')
    folder = Path(job['output_path']).resolve()
    if folder.parent != DATASETS.resolve():
        raise ValueError('数据集路径无效。')
    path = folder.with_suffix('.zip')
    temporary = folder.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for file in sorted(folder.rglob('*')):
                if file.is_file():
                    with file.open('rb') as source, archive.open(file.relative_to(folder).as_posix(), 'w', force_zip64=True) as target:
                        while chunk := source.read(1024 * 1024):
                            if stop.is_set():
                                raise ValueError('打包已取消，数据集目录仍然保留。')
                            target.write(chunk)
        temporary.replace(path)
        return assets.add(path, path.name, 'output', output_type='dataset', dataset_path=str(folder))
    finally:
        temporary.unlink(missing_ok=True)


batches = BatchStore()
