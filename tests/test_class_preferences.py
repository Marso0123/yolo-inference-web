import importlib

import pytest

from yoloview.preferences import ClassPreferences


def test_choices_survive_restart_and_remain_per_model(tmp_path):
    path = tmp_path / 'preferences.json'
    preferences = ClassPreferences(path)
    names = {0: 'person', 5: 'bus'}
    preferences.save('first-weight', names, [5, 5])
    preferences.save('second-weight', names, [0])
    reopened = ClassPreferences(path)
    assert reopened.get('first-weight', names) == [5]
    assert reopened.get('second-weight', names) == [0]
    assert reopened.get('never-configured', names) == []
    reopened.save('first-weight', names, [])  # Empty means the default: all classes.
    assert ClassPreferences(path).get('first-weight', names) == []
    assert ClassPreferences(path).get('second-weight', names) == [0]


def test_invalid_ids_or_remapped_labels_do_not_silently_apply(tmp_path):
    preferences = ClassPreferences(tmp_path / 'preferences.json')
    names = {0: 'person', 5: 'bus'}
    preferences.save('weight', names, [5])
    for invalid in ([100], [True], ['5']):
        with pytest.raises(ValueError, match='类别编号'):
            preferences.save('weight', names, invalid)
    assert preferences.get('weight', names) == [5]
    with pytest.raises(ValueError, match='类别映射'):
        preferences.get('weight', {0: 'bus', 5: 'person'})


def test_model_identity_uses_content_instead_of_filename(tmp_path, monkeypatch):
    storage = importlib.import_module('yoloview.storage')
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setattr(storage, 'DATA', data)
    assets = storage.Assets()
    keys = []
    for folder, content in [('one', b'first trained weights'), ('two', b'other trained weights'),
                            ('three', b'first trained weights')]:
        directory = data / folder
        directory.mkdir()
        path = directory / 'best.pt'
        path.write_bytes(content)
        item = assets.add(path, 'best.pt', 'model')
        keys.append(assets.model_key(item['id']))
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]
