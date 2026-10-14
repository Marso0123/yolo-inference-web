from __future__ import annotations

import ast
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

from .storage import ROOT

# Preload the CUDA/cuDNN libraries packaged with PyTorch, including on Windows.
ort.preload_dlls()
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def hardware():
    return dict(gpu_available=torch.cuda.is_available(),
                gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                torch=torch.__version__, ultralytics=__import__('ultralytics').__version__,
                onnxruntime=ort.__version__, providers=ort.get_available_providers())


class Runner:
    def __init__(self, path, options):
        self.path = Path(path)
        self.options = dict(options)
        self.device = '0' if options['device'] == 'gpu' else 'cpu'
        if self.device == '0' and not torch.cuda.is_available():
            raise ValueError('GPU 不可用，请检查 NVIDIA 驱动或选择 CPU。')
        self.size = options['imgsz']
        if self.path.suffix.lower() == '.onnx':
            graph = onnx.load(str(self.path), load_external_data=False)
            metadata = {m.key: m.value for m in graph.metadata_props}
            task = metadata.get('task')
            if task not in ('detect', 'segment') or 'names' not in metadata:
                raise ValueError('ONNX 缺少检测／分割任务或类别元数据，请使用本工具从 PT 导出。')
            ast.literal_eval(metadata['names'])
            dims = graph.graph.input[0].type.tensor_type.shape.dim
            if dims[0].dim_value not in (0, 1):
                raise ValueError('只支持 batch=1 的 ONNX，请重新导出。')
            h, w = dims[2].dim_value, dims[3].dim_value
            if h and w:
                self.size = (h, w)
            self.model = YOLO(str(self.path), task=task)
        else:
            self.model = YOLO(str(self.path))
        if self.model.task not in ('detect', 'segment'):
            raise ValueError('第一版仅支持目标检测和实例分割模型。')
        kwargs = dict(device=self.device, imgsz=self.size, verbose=False, save=False, rect=False, quantize=32, nms=None)
        # Build before inference so a failed GPU backend cannot silently execute on CPU.
        self.model.predictor = self.model._smart_load('predictor')(overrides=kwargs, _callbacks=self.model.callbacks)
        self.model.predictor.setup_model(model=self.model.model, verbose=False)
        backend = self.model.predictor.model
        self.cuda_nodes = None
        if self.path.suffix.lower() == '.onnx':
            if self.device == '0':
                from ultralytics.nn.backends.onnx import ONNXBackend
                session_options = ort.SessionOptions()
                session_options.enable_profiling = True
                session_options.profile_file_prefix = str(ROOT / '.runtime' / 'profiles' / 'cuda')
                replacement = ONNXBackend(str(self.path), torch.device('cuda:0'), session_options=session_options)
                backend.backend = replacement
            backend.session.disable_fallback()
            if self.device == '0' and 'CUDAExecutionProvider' not in backend.session.get_providers():
                raise ValueError('ONNX CUDA 初始化失败，已停止；请检查 CUDA／cuDNN 环境或选择 CPU。')
        elif self.device == '0' and next(backend.model.parameters()).device.type != 'cuda':
            raise ValueError('PT 模型未加载到 GPU，已停止。')
        size = (self.size, self.size) if isinstance(self.size, int) else self.size
        self.model.predict(np.zeros((*size, 3), dtype=np.uint8), **kwargs)
        if self.path.suffix.lower() == '.onnx' and self.device == '0':
            profile = Path(backend.session.end_profiling())
            events = json.loads(profile.read_text('utf-8'))
            self.cuda_nodes = sum(e.get('args', {}).get('provider') == 'CUDAExecutionProvider' for e in events)
            profile.unlink(missing_ok=True)
            if not self.cuda_nodes:
                raise ValueError('ONNX 实际推理未执行 CUDA 节点，已停止。')
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        selected = options.get('classes')
        if selected and any(i not in self.names for i in selected):
            raise ValueError('类别过滤编号不属于当前模型，请重新选择类别。')
        self.info = dict(task=self.model.task, names=self.names, device='GPU / CUDA' if self.device == '0' else 'CPU',
                         imgsz=self.size, cuda_nodes=self.cuda_nodes,
                         end2end=bool(getattr(backend, 'end2end', False)))

    def predict(self, frame):
        result, elapsed = self.predict_result(frame)
        return self.render_result(result, elapsed)

    def predict_result(self, frame):
        """Return raw results internally for dataset export, never as API coordinates."""
        started = time.perf_counter()
        result = self.model.predict(frame, device=self.device, imgsz=self.size, rect=False, quantize=32, nms=None,
                                    conf=self.options['conf'], iou=self.options['iou'],
                                    classes=self.options.get('classes') or None,
                                    verbose=False, save=False)[0]
        if self.device == '0':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return result, elapsed

    def render_result(self, result, elapsed):
        plotted = result.plot(boxes=self.options['boxes'], masks=self.options['masks'],
                              labels=self.options['labels'], conf=self.options['labels'], pil=False)
        counts = {}
        if result.boxes is not None:
            for cls in result.boxes.cls.cpu().numpy().astype(int):
                label = self.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1
        # Never return boxes or masks through the public API.
        return plotted, dict(inference_ms=round(elapsed, 1), count=sum(counts.values()), counts=counts)

    def close(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
