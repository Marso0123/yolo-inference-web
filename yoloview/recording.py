from __future__ import annotations

import math
import queue
import subprocess
import threading
import time

import cv2

from .storage import ROOT, assets, output_name


class Recorder:
    """Write off the inference thread; source video frames are never resampled."""

    def __init__(self, frame, fps=25, realtime=True, source_index=None):
        self.output = output_name('标注视频', '.mp4')
        self.temporary = self.output.with_suffix('.avi')
        self.fps = float(fps) if math.isfinite(fps) and 0 < fps <= 240 else 25.0
        self.realtime = realtime
        self.source_index = source_index
        self.last_source_index = source_index
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.closed = False
        self.error = None
        self.queue = queue.Queue(maxsize=8)
        self.frames_written = 0
        h, w = frame.shape[:2]
        self.size = (w + w % 2, h + h % 2)
        self.writer = cv2.VideoWriter(str(self.temporary), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, self.size)
        if not self.writer.isOpened():
            self.writer.release()
            raise ValueError('无法创建录制文件，请检查磁盘空间。')
        self.queue.put((frame, 0.0))
        self.thread = threading.Thread(target=self._run, daemon=True, name='video-writer')
        self.thread.start()

    def _put(self, item):
        while True:
            if self.error:
                raise ValueError('视频写入失败，临时 AVI 已保留：' + str(self.error))
            if not self.thread.is_alive():
                raise ValueError('视频写入线程已退出，临时 AVI 已保留。')
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                # Bounded memory and backpressure, without holding the Engine state lock.
                continue

    def write(self, frame, now=None, source_index=None):
        with self.lock:
            if self.closed:
                return False
            if not self.realtime:
                if source_index is None or self.last_source_index is None:
                    raise ValueError('本地视频录制缺少原始帧编号。')
                if source_index <= self.last_source_index:
                    return False
                if source_index != self.last_source_index + 1:
                    raise ValueError('录制帧不连续，已停止录制，请重新开始。')
            elapsed = max(0.0, (time.monotonic() if now is None else now) - self.started)
            self._put((frame, elapsed))
            if not self.realtime:
                self.last_source_index = source_index
            return True

    def _write(self, frame):
        self.writer.write(frame)
        self.frames_written += 1

    def _run(self):
        previous = None
        try:
            while True:
                item = self.queue.get()
                if item[0] is None:
                    if self.realtime and previous is not None:
                        end = max(1, math.ceil(item[1] * self.fps - 1e-7))
                        while self.frames_written < end:
                            self._write(previous)
                    break
                frame, elapsed = item
                current = cv2.resize(frame, self.size)
                if self.realtime:
                    target = int(elapsed * self.fps + 1e-7)
                    if self.frames_written > target:
                        previous = current
                        continue
                    while previous is not None and self.frames_written < target:
                        self._write(previous)
                self._write(current)
                previous = current
        except Exception as error:
            self.error = error
        finally:
            self.writer.release()

    def finish(self):
        with self.lock:
            if self.closed:
                raise ValueError('该录制已结束。')
            self.closed = True
            elapsed = max(0.0, time.monotonic() - self.started)
            try:
                self._put((None, elapsed))
            except Exception:
                self.thread.join(timeout=5)
                raise
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise ValueError('视频写入尚未结束，临时 AVI 已保留。')
        if self.error:
            raise ValueError('视频写入失败，临时 AVI 已保留：' + str(self.error))
        ffmpeg = ROOT / '.runtime' / 'ffmpeg.exe'
        if not ffmpeg.exists():
            raise ValueError('缺少 FFmpeg，原始录制文件已保留，请重新运行安装脚本。')
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(self.temporary),
                   '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(self.output)]
        completed = subprocess.run(command, capture_output=True, timeout=600, creationflags=creationflags)
        if completed.returncode:
            raise ValueError('视频编码失败，原始 AVI 已保留：' + completed.stderr.decode('utf-8', errors='replace')[-500:])
        self.temporary.unlink(missing_ok=True)
        return assets.add(self.output, self.output.name, 'output', output_type='video',
                          fps=self.fps, frames=self.frames_written,
                          duration=round(self.frames_written / self.fps, 3),
                          recording_mode='realtime' if self.realtime else 'source_frames')
