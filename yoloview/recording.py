from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2

from .storage import ROOT, assets, output_name


class Recorder:
    """Sample annotations at 20 FPS using elapsed time, holding the previous result."""

    def __init__(self, frame):
        self.output = output_name('标注视频', '.mp4')
        self.temporary = self.output.with_suffix('.avi')
        self.fps = 20
        self.started = time.monotonic()
        self.index = 0
        h, w = frame.shape[:2]
        self.size = (w + w % 2, h + h % 2)
        self.previous = cv2.resize(frame, self.size)
        self.writer = cv2.VideoWriter(str(self.temporary), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, self.size)
        if not self.writer.isOpened():
            raise ValueError('无法创建录制文件，请检查磁盘空间。')
        self.writer.write(self.previous)
        self.index = 1

    def write(self, frame, now=None):
        now = time.monotonic() if now is None else now
        due = int(max(0, now - self.started) * self.fps) + 1
        while self.index < due:
            self.writer.write(self.previous)
            self.index += 1
        self.previous = cv2.resize(frame, self.size)

    def finish(self):
        self.write(self.previous)
        self.writer.release()
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
        return assets.add(self.output, self.output.name, 'output', output_type='video')
