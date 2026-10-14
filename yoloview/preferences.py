from __future__ import annotations

import json
import threading
from pathlib import Path

from .storage import DATA


class ClassPreferences:
    """Store user choices by weight content, with the model's exact class mapping."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()

    def read(self):
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text('utf-8'))
        except (ValueError, OSError) as error:
            raise ValueError('类别配置文件读取失败，请检查 data/class_preferences.json。') from error
        if not isinstance(data, dict):
            raise ValueError('类别配置文件格式不正确。')
        return data

    @staticmethod
    def mapping(names):
        return {str(k): str(v) for k, v in names.items()}

    def get(self, key, names):
        with self.lock:
            saved = self.read().get(key)
            if saved is None:
                return []
            if not isinstance(saved, dict) or saved.get('names') != self.mapping(names):
                raise ValueError('模型类别映射与已保存的配置不一致，请确认模型文件和类别配置。')
            classes = saved.get('classes')
            if not isinstance(classes, list) or any(type(c) is not int or c not in names for c in classes):
                raise ValueError('已保存的类别配置包含无效类别，请检查配置文件。')
            return sorted(set(classes))

    def save(self, key, names, classes):
        if any(type(c) is not int or c not in names for c in classes):
            raise ValueError('类别编号不属于当前模型。')
        classes = sorted(set(classes))
        with self.lock:
            data = self.read()
            data[key] = dict(names=self.mapping(names), classes=classes)
            temporary = self.path.with_suffix('.tmp')
            try:
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
                temporary.replace(self.path)
            except OSError as error:
                raise ValueError('类别配置保存失败，请检查磁盘空间和文件权限。') from error
        return classes


class_preferences = ClassPreferences(DATA / 'class_preferences.json')
