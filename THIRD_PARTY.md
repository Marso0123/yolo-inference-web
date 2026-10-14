# 第三方组件与来源

项目代码调用第三方库，具体安装版本见 `requirements.lock.txt`。

| 组件 | 主要用途 | 上游来源与许可 |
|---|---|---|
| Ultralytics | YOLO 检测、实例分割与导出 | https://github.com/ultralytics/ultralytics / AGPL-3.0，另有上游商业许可 |
| PyTorch / TorchVision | PT 模型 CPU／CUDA 执行 | https://pytorch.org / BSD 类许可，参见发行包许可 |
| ONNX / ONNX Runtime | ONNX 图检查与执行 | https://onnx.ai / https://github.com/microsoft/onnxruntime / Apache-2.0 与 MIT |
| OpenCV | 图像、视频解码与录制 | https://opencv.org / Apache-2.0，参见发行包第三方声明 |
| FastAPI / Uvicorn | 本机网页后端 | https://github.com/fastapi/fastapi / https://github.com/encode/uvicorn |
| FFmpeg | 将临时录制编码为 H.264 MP4 | https://ffmpeg.org / Windows 构建来自 https://www.gyan.dev/ffmpeg/builds/ |
| MediaMTX | 仅本次本机 RTSP 验证 | https://github.com/bluenviron/mediamtx / MIT |

当前 `.runtime/ffmpeg.exe` 是 gyan.dev 8.1.1 full 构建，构建参数包含 GPL 与 version3；重新安装脚本使用发布者的 essentials 构建。FFmpeg 上游许可与源代码说明：https://ffmpeg.org/legal.html 。

验证 nano 权重来自 Ultralytics 官方 assets releases，测试 bus 图片来自 https://ultralytics.com/images/bus.jpg 。安全帽 `best.pt` 和图片从用户本机目录复制，用于本地功能检查。软件、预训练权重和数据的许可分别以各自上游或权利人的声明为准。

README 的 `docs/screenshots/` 图片为本工具实际运行页面截图，演示使用官方 YOLO11n／YOLO26n 模型、上述公开 bus 示例图片及生成的空白图；不包含用户自训练模型、私人图片或完整本机输出路径。截图中的示例图片与模型仍以各自上游许可为准。
