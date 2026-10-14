from __future__ import annotations

import json
import hashlib
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUTS = DATA / 'outputs'
MODELS = DATA / 'models'
MEDIA = DATA / 'media'
for directory in (OUTPUTS, MODELS, MEDIA, ROOT / 'logs', ROOT / '.runtime' / 'profiles'):
    directory.mkdir(parents=True, exist_ok=True)


class Assets:
    """Persist only uploaded-file identities; never export inference coordinates."""

    def __init__(self):
        self.lock = threading.RLock()
        self.file = DATA / 'assets.json'
        self.items = json.loads(self.file.read_text('utf-8')) if self.file.exists() else {}
        for item in self.items.values():
            path = Path(item['path'])
            if path.is_absolute() and path.is_relative_to(ROOT):
                item['path'] = path.relative_to(ROOT).as_posix()

    def add(self, path: Path, name: str, kind: str, **metadata):
        with self.lock:
            key = uuid.uuid4().hex
            item = dict(id=key, name=name, kind=kind, path=path.resolve().relative_to(ROOT).as_posix(), **metadata)
            self.items[key] = item
            temporary = self.file.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), 'utf-8')
            temporary.replace(self.file)
            return self.public(item)

    def model_key(self, key):
        """Hash immutable uploaded weights once; identical uploads share preferences."""
        with self.lock:
            item = self.get(key, 'model')
            path = Path(item['path'])
            stat = path.stat()
            if item.get('sha256') and item.get('hash_stat') == [stat.st_size, stat.st_mtime_ns]:
                return item['sha256']
            digest = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            result = digest.hexdigest()
            self.items[key].update(sha256=result, hash_stat=[stat.st_size, stat.st_mtime_ns])
            temporary = self.file.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), 'utf-8')
            temporary.replace(self.file)
            return result

    @staticmethod
    def public(item):
        return {k: v for k, v in item.items() if k != 'path'}

    def get(self, key, kind=None):
        with self.lock:
            item = self.items.get(key)
            if not item or (kind and item['kind'] != kind) or not (ROOT / item['path']).is_file():
                raise ValueError('文件不存在，请重新选择或上传。')
            return dict(item, path=str(ROOT / item['path']))

    def list(self, kind=None):
        with self.lock:
            return [self.public(v) for v in self.items.values()
                    if (kind is None or v['kind'] == kind) and (ROOT / v['path']).is_file()]


def safe_name(name):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name.replace('\\', '/').split('/')[-1])[:160] or 'file'


def output_name(prefix, suffix):
    return OUTPUTS / f'{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}{suffix}'


assets = Assets()
