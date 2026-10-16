from __future__ import annotations

import gc
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import onnx
from ultralytics import YOLO

from .inference import Runner
from .batch import batches, run_dataset, archive_dataset
from .recording import Recorder
from .preferences import class_preferences
from .storage import MODELS, ROOT, assets, output_name

LOGGER = logging.getLogger('yoloview')


class LiveSource:
    """Read continuously into one slot. Reset the slot and epoch on disconnection."""

    def __init__(self, url, stop, disconnected):
        self.url = url
        self.stop = stop
        self.disconnected = disconnected
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.epoch = 0
        self.fps = 25.0
        self.thread = threading.Thread(target=self.read, daemon=True, name='rtsp-decoder')
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.frame, self.seq, self.epoch

    def read(self):
        while not self.stop.is_set():
            capture = None
            try:
                capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                          [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                if not capture.isOpened():
                    raise ValueError('连接失败')
                fps = capture.get(cv2.CAP_PROP_FPS)
                self.fps = float(fps) if np.isfinite(fps) and 0 < fps <= 240 else 25.0
                while not self.stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError('视频流断开')
                    with self.lock:
                        self.frame = frame
                        self.seq += 1
            except Exception:
                with self.lock:
                    self.frame = None
                    self.epoch += 1
                if not self.stop.is_set():
                    self.disconnected()
                    self.stop.wait(2)
            finally:
                if capture is not None:
                    capture.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('RTSP 解码线程尚未退出，请稍后重试。')


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.control = threading.RLock()
        self.runner = None
        self.loaded_id = None
        self.loaded_key = None
        self.thread = None
        self.stop_event = threading.Event()
        self.recorder = None
        self.record_saving = False
        self.record_finish_lock = threading.RLock()
        self.record_result = None
        self.seq = 0
        self.generation = 0
        self.jpeg = None
        self.frame = None
        self.frame_index = None
        self.state = dict(status='idle', message='请选择模型', source_kind=None, model=None, recording=False,
                          saving=False, fps=0, inference_ms=0, count=0, counts={}, position=None,
                          last_saved=None, export=None, record_error=None, batch=None,
                          source_fps=None, recording_fps=None, recording_mode=None,
                          record_armed=False, recording_result=None)

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def status(self):
        with self.lock:
            return dict(self.state, generation=self.generation, seq=self.seq, has_frame=self.jpeg is not None)

    def clear_frame(self):
        with self.lock:
            self.jpeg = None
            self.frame = None
            self.frame_index = None
            self.seq += 1
            self.state.update(count=0, counts={}, fps=0, inference_ms=0, position=None, source_fps=None)

    def snapshot(self):
        with self.lock:
            return self.status(), self.jpeg

    def ensure_idle(self):
        if self.thread and self.thread.is_alive():
            raise ValueError('请先停止当前推理，或等待转换／数据集打包完成。')
        if self.record_saving:
            raise ValueError('正在保存视频，请等待完成。')

    def load(self, model_id, options):
        with self.control:
            self.ensure_idle()
            item = assets.get(model_id, 'model')
            self.clear_frame()
            self.update(status='loading', message='正在加载模型并验证推理设备…', model=None, export=None)
            if self.runner:
                self.runner.close()
            self.runner = None
            self.loaded_id = None
            self.loaded_key = None
            runner = None
            try:
                runner = Runner(item['path'], {**options, 'classes': None})
                model_key = assets.model_key(model_id)
                selected = class_preferences.get(model_key, runner.names)
                runner.options['classes'] = selected or None
                self.runner = runner
                self.loaded_id = model_id
                self.loaded_key = model_key
                info = dict(runner.info, name=item['name'], id=model_id, classes=selected)
                self.update(status='ready', message='模型已就绪', model=info, source_kind=None)
                return info
            except Exception as error:
                LOGGER.exception('Model load failed')
                if runner:
                    runner.close()
                gc.collect()
                self.update(status='error', message=str(error))
                raise ValueError(str(error)) from error

    def save_class_preferences(self, model_id, classes):
        with self.control:
            self.ensure_idle()
            if not self.runner or model_id != self.loaded_id:
                raise ValueError('请先加载当前选中的模型。')
            selected = class_preferences.save(self.loaded_key, self.runner.names, classes)
            with self.lock:
                self.state['model'] = dict(self.state['model'], classes=selected)
            # Keep the active result unchanged; Engine.start applies the next selection.
            return dict(model_id=model_id, classes=selected)

    def start(self, request):
        with self.control:
            self.ensure_idle()
            if not self.runner or request['model_id'] != self.loaded_id:
                raise ValueError('请先加载当前选中的模型。')
            options = request['options']
            if options['device'] != self.runner.options['device']:
                raise ValueError('设备已改变，请重新加载模型。')
            if options['imgsz'] != self.runner.options['imgsz']:
                raise ValueError('输入尺寸已改变，请重新加载模型。')
            selected = options.get('classes')
            if selected and any(c not in self.runner.names for c in selected):
                raise ValueError('类别编号不属于当前模型。')
            self.runner.options.update(options)
            kind = request['kind']
            save_full = request.get('save_full_video', False)
            if save_full and kind != 'video':
                raise ValueError('全程录制仅支持本地视频。')
            if kind == 'rtsp':
                source = request['url'].strip()
                if not source.lower().startswith(('rtsp://', 'rtsps://')):
                    raise ValueError('请输入 rtsp:// 或 rtsps:// 开头的地址。')
            else:
                source = assets.get(request['media_id'], 'media')
                if source['media_type'] != kind:
                    raise ValueError('选择的文件类型与输入类型不一致。')
                source = source['path']
            self.generation += 1
            self.clear_frame()
            self.stop_event = threading.Event()
            self.record_result = None
            self.update(status='running', source_kind=kind, message='正在推理', export=None,
                        record_armed=save_full, recording_result=None, record_error=None)
            self.thread = threading.Thread(target=self.run, args=(kind, source), daemon=True, name='inference')
            self.thread.start()
            return self.status()

    def disconnected(self):
        self.clear_frame()
        self.update(status='reconnecting', message='RTSP 连接中断，正在自动重连…')
        # A disconnected stream cannot keep recording an old result.
        self.finish_recording()

    def start_batch(self, request):
        with self.control:
            self.ensure_idle()
            if not self.runner or request['model_id'] != self.loaded_id:
                raise ValueError('请先加载当前选中的模型。')
            options = request['options']
            if any(options[k] != self.runner.options[k] for k in ('device', 'imgsz')):
                raise ValueError('设备或输入尺寸已改变，请重新加载模型。')
            selected = options.get('classes')
            if selected and any(c not in self.runner.names for c in selected):
                raise ValueError('类别编号不属于当前模型。')
            key = request['batch_id']
            with batches.lock:
                job = batches.get(key)
                if job['status'] != 'uploading' or not job['items']:
                    raise ValueError('请选择并上传图片；每个批量任务只能运行一次。')
                job = batches.update(key, status='running', message='正在批量推理',
                                     model=self.state['model'], remap=request['remap'], preview=request['preview'])
            self.runner.options.update(options)
            self.generation += 1
            self.clear_frame()
            self.stop_event = threading.Event()
            self.update(status='running', source_kind='batch', message='正在批量推理', export=None, batch=job)
            self.thread = threading.Thread(target=self.run_batch, args=(request,), daemon=True, name='batch-inference')
            self.thread.start()
            return self.status()

    def run_batch(self, request):
        key = request['batch_id']
        try:
            job = run_dataset(batches, key, self.runner, self.stop_event, self.publish,
                              lambda job: self.update(batch=job), self.state['model'],
                              request['remap'], request['preview'])
            self.update(status='finished', batch=job,
                        message=f"{job['message']}：保存 {job['done']} 张，失败 {job['failed']} 张，待复核 {job['review']} 张")
        except Exception as error:
            LOGGER.exception('Batch failed')
            job = batches.update(key, status='failed', message=str(error))
            self.update(status='error', batch=job, message='批量任务失败：' + str(error))

    def archive_batch(self, key):
        with self.control:
            self.ensure_idle()
            job = batches.get(key)
            if job['status'] not in ('completed', 'completed_with_errors', 'cancelled', 'failed', 'interrupted') or not job.get('output_path'):
                raise ValueError('请等待批量任务结束并生成数据集后再打包。')
            if job.get('archive'):
                assets.get(job['archive']['id'], 'output')
                return self.status()
            self.stop_event = threading.Event()
            current = batches.update(key, status='packaging', message='正在生成 ZIP，数据集目录保留')
            self.update(status='packaging', batch=current, message='正在打包数据集…')
            self.thread = threading.Thread(target=self.run_archive, args=(key, job['status']), daemon=True, name='dataset-zip')
            self.thread.start()
            return self.status()

    def run_archive(self, key, prior):
        try:
            item = archive_dataset(batches, key, self.stop_event)
            job = batches.update(key, status=prior, archive=item, message='ZIP 已生成，可下载；数据集目录保留')
            self.update(status='finished', batch=job, message=job['message'], last_saved=item)
        except Exception as error:
            LOGGER.exception('Dataset archive failed')
            job = batches.update(key, status=prior, message=str(error))
            self.update(status='finished' if self.stop_event.is_set() else 'error', batch=job, message=str(error))

    def publish(self, frame, stats, position=None, frame_index=None, recording=None):
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError('预览图像编码失败。')
        with self.lock:
            self.frame = frame
            self.frame_index = frame_index
            self.jpeg = jpeg.tobytes()
            self.seq += 1
            recorder = self.recorder
            if frame_index is not None and recorder is not recording:
                recorder = None
            message = '正在逐帧推理并录制，保存时保持源视频帧率' if self.recorder and not self.recorder.realtime else '正在推理'
            self.state.update(stats, position=position)
            if self.state['status'] not in ('finalizing', 'stopping'):
                self.state.update(status='running')
                if not self.record_saving and not self.state['record_error']:
                    self.state['message'] = message
        # MJPEG compression / queue backpressure must not hold the preview/status lock.
        if recorder:
            try:
                recorder.write(frame, source_index=frame_index)
            except Exception as error:
                LOGGER.exception('Recording write failed')
                self.finish_recording(expected=recorder)
                self.update(record_error=str(error), message='录制已停止：' + str(error))

    def run(self, kind, source):
        capture = None
        live = None
        started = time.monotonic()
        last_published = None
        active_recording = None
        completed = False
        try:
            if kind == 'image':
                frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('图片无法解码，请选择 JPG／PNG／BMP／WebP。')
                result, stats = self.runner.predict(frame)
                if not self.stop_event.is_set():
                    self.publish(result, dict(stats, fps=0))
                    self.update(status='finished', message='图片推理完成')
                return
            if kind == 'video':
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    raise ValueError('视频无法解码，请尝试 MP4（H.264）格式。')
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if np.isfinite(fps) and 0 < fps <= 240 else 25
                self.update(source_fps=fps)
                count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = count / fps if np.isfinite(count) and count > 0 else None
                next_index = 0
            else:
                live = LiveSource(source, self.stop_event, self.disconnected)
                last_seq = -1
            while not self.stop_event.is_set():
                position = None
                frame_index = None
                recording = None
                if live:
                    frame, seq, epoch = live.snapshot()
                    if frame is None or seq == last_seq:
                        self.stop_event.wait(0.01)
                        continue
                    last_seq = seq
                    self.update(source_fps=live.fps)
                else:
                    with self.lock:
                        recording = self.recorder
                    if recording is not active_recording:
                        if recording:
                            # Start from the last displayed frame, even if a newer preview
                            # was already in flight when the user pressed Record.
                            next_index = recording.source_index + 1
                            if not capture.set(cv2.CAP_PROP_POS_FRAMES, next_index):
                                raise ValueError('无法定位录制起点，请尝试 MP4（H.264）视频。')
                            if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - next_index) > 0.5:
                                raise ValueError('视频未能准确定位录制起点，已停止以避免跳帧。')
                        active_recording = recording
                        # Resume preview from the current source time after a slow recording.
                        started = time.monotonic() - next_index / fps
                    elapsed = time.monotonic() - started
                    target = int(elapsed * fps)
                    # Ordinary preview can skip stale frames; source recording cannot.
                    while (not recording and not self.state['record_armed'] and next_index < target
                           and not self.stop_event.is_set()):
                        if not capture.grab():
                            break
                        next_index += 1
                    if self.stop_event.is_set():
                        break
                    ok, frame = capture.read()
                    if not ok:
                        completed = True
                        break
                    position = dict(seconds=round(next_index / fps, 2), duration=duration)
                    frame_index = next_index
                    next_index += 1
                result, stats = self.runner.predict(frame)
                if self.stop_event.is_set():
                    break
                if capture:
                    with self.lock:
                        if self.state['record_armed']:
                            self._begin_recording(result, fps, False, frame_index)
                            recording = self.recorder
                            # Already seeded with this first result; next iteration starts at frame 1.
                            active_recording = recording
                            started = time.monotonic() - next_index / fps
                        if self.recorder is not None and self.recorder is not recording:
                            continue
                now = time.monotonic()
                preview_fps = 0 if last_published is None else 1 / max(0.001, now - last_published)
                last_published = now
                if live:
                    # Serialize the epoch check with decoder invalidation, including publication.
                    with live.lock:
                        if live.epoch != epoch:
                            continue
                        self.publish(result, dict(stats, fps=round(preview_fps, 1)), position)
                else:
                    self.publish(result, dict(stats, fps=round(preview_fps, 1)), position,
                                 frame_index=frame_index, recording=recording)
                if capture:
                    self.stop_event.wait(max(0, next_index / fps - (time.monotonic() - started)))
        except Exception as error:
            LOGGER.exception('Inference failed')
            self.clear_frame()
            self.update(status='error', message=str(error))
        finally:
            if capture:
                capture.release()
            if live:
                live.close()
            with self.lock:
                had_recording = bool(self.recorder or self.record_saving)
                self.state['record_armed'] = False
                if had_recording and self.state['status'] != 'error':
                    self.state.update(status='finalizing', message='推理已结束，正在保存录制视频…')
            self.finish_recording()
            with self.lock:
                if self.state['status'] != 'error' and (completed or self.stop_event.is_set()):
                    self.state['status'] = 'finished' if completed else 'stopped'
                    ending = '视频推理完成' if completed else '推理已停止'
                    if self.state['record_error']:
                        self.state['message'] = ending + '；录制保存失败：' + self.state['record_error']
                    else:
                        self.state['message'] = ending + ('，录制视频已保存' if self.record_result else '')

    def stop(self):
        with self.control:
            self.stop_event.set()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=15)
                if self.thread.is_alive():
                    self.update(status='stopping', message='正在等待当前操作退出…')
                    raise ValueError('当前操作尚未退出，请稍后再次点击停止。')
            self.finish_recording()
            if self.runner and self.state['status'] not in ('error', 'finished', 'stopped'):
                self.update(status='stopped', message='推理已停止')
            return self.status()

    def save_image(self):
        with self.lock:
            if self.frame is None:
                raise ValueError('当前没有可保存的推理结果。')
            frame = self.frame.copy()
        path = output_name('标注图片', '.jpg')
        ok, image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError('图片编码失败。')
        image.tofile(str(path))
        item = assets.add(path, path.name, 'output', output_type='image')
        self.update(last_saved=item)
        return item

    def start_recording(self):
        with self.lock:
            if self.state['status'] != 'running' or self.state['source_kind'] not in ('video', 'rtsp'):
                raise ValueError('请在视频或 RTSP 正在推理时开始录制。')
            if self.frame is None:
                raise ValueError('请等待第一帧推理结果。')
            if self.recorder or self.record_saving or self.state['record_armed']:
                raise ValueError('已有录制或保存任务。')
            realtime = self.state['source_kind'] == 'rtsp'
            if not realtime and self.frame_index is None:
                raise ValueError('请等待视频帧就绪。')
            fps = self.state['source_fps'] or 25
            self._begin_recording(self.frame, fps, realtime, self.frame_index)
            return self.status()

    def _begin_recording(self, frame, fps, realtime, source_index):
        """Caller holds state lock; also used to seed full-video recording at frame zero."""
        self.recorder = Recorder(frame, fps=fps, realtime=realtime, source_index=source_index)
        self.record_result = None
        self.state.update(recording=True, record_armed=False, record_error=None,
                          recording_result=None, recording_fps=fps,
                          recording_mode='realtime' if realtime else 'source_frames',
                          message='正在录制实时推理画面' if realtime else '正在逐帧录制，保存时保持源视频帧率')

    def finish_recording(self, strict=False, expected=None):
        # Concurrent manual stop / EOF callers wait for the same encoding result.
        # No Engine state lock is held during encoding or while waiting here.
        with self.record_finish_lock:
            return self._finish_recording(strict, expected)

    def _finish_recording(self, strict=False, expected=None):
        with self.lock:
            recorder = self.recorder
            if expected is not None and recorder is not expected:
                return None
            if recorder is None:
                if strict and self.state['record_error']:
                    raise ValueError(self.state['record_error'])
                return self.record_result if strict else None
            self.recorder = None
            self.record_saving = True
            self.state.update(recording=False, saving=True, recording_fps=None, recording_mode=None)
            if self.state['status'] != 'error':
                self.state['message'] = ('推理已结束，正在保存录制视频…' if self.state['status'] == 'finalizing'
                                         else '正在编码保存录制视频…')
        try:
            item = recorder.finish()
            with self.lock:
                self.record_result = item
                message = '录制视频已保存：' + item.get('name', '标注视频')
                if self.state['status'] == 'error':
                    message = self.state['message'] + '；' + message
                self.state.update(last_saved=item, recording_result=item, message=message)
            return item
        except Exception as error:
            LOGGER.exception('Recording failed')
            with self.lock:
                message = '录制保存失败：' + str(error)
                if self.state['status'] == 'error':
                    message = self.state['message'] + '；' + message
                self.state.update(message=message, record_error=str(error))
            if strict:
                raise ValueError(str(error)) from error
            return None
        finally:
            with self.lock:
                self.record_saving = False
                self.state['saving'] = False

    def export(self, model_id, size):
        with self.control:
            self.ensure_idle()
            item = assets.get(model_id, 'model')
            if Path(item['path']).suffix.lower() != '.pt':
                raise ValueError('请选择 PT 模型进行转换。')
            if self.runner:
                self.runner.close()
            self.runner = None
            self.loaded_id = None
            self.clear_frame()
            self.update(status='exporting', message='正在转换为 ONNX…', model=None, export=dict(status='running'))
            self.thread = threading.Thread(target=self.run_export, args=(item, size), daemon=True, name='onnx-export')
            self.thread.start()
            return self.status()

    def run_export(self, item, size):
        folder = MODELS / uuid.uuid4().hex
        folder.mkdir()
        copied = folder / Path(item['path']).name
        try:
            shutil.copy2(item['path'], copied)
            model = YOLO(str(copied))
            if model.task not in ('detect', 'segment'):
                raise ValueError('仅支持检测和实例分割 PT 转换。')
            self.update(message='正在导出 ONNX，首次转换可能需要稍等…')
            # FP32, fixed shape, external postprocessing; opset 17 works with the pinned ORT.
            exported = Path(model.export(format='onnx', imgsz=size, batch=1, device='cpu',
                                         quantize=32, dynamic=False, simplify=False, opset=17, nms=None))
            onnx.checker.check_model(str(exported))
            model = None
            self.update(message='ONNX 已生成，正在执行实际推理检查…')
            validation = Runner(exported, default_options(imgsz=size))
            frame, _ = validation.predict(np.zeros((size, size, 3), dtype=np.uint8))
            if frame is None:
                raise ValueError('导出的 ONNX 未通过推理检查。')
            validation.close()
            result = assets.add(exported, Path(item['name']).stem + f'_{size}.onnx', 'model', exported=True)
            self.update(status='ready', message='转换完成，已通过 CPU 实际推理检查；可加载查看效果',
                        export=dict(status='done', result=result, validation='CPU 实际推理通过'))
        except Exception as error:
            LOGGER.exception('Export failed')
            self.update(status='error', message='转换失败：' + str(error), export=dict(status='error'))
        finally:
            copied.unlink(missing_ok=True)
            gc.collect()


def default_options(**changes):
    return {**dict(device='cpu', imgsz=640, conf=0.25, iou=0.45, classes=None,
                   boxes=True, masks=True, labels=True), **changes}


engine = Engine()
