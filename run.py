import os

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(os.path.dirname(__file__), '.runtime', 'ultralytics'))
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=8790, log_level='info', access_log=False)
