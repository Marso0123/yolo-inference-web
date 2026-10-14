"""Install one FFmpeg executable from the publisher's official Windows archive."""
import shutil
import urllib.request
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runtime = root / '.runtime'
runtime.mkdir(exist_ok=True)
target = runtime / 'ffmpeg.exe'
if not target.exists():
    archive = runtime / 'ffmpeg-essentials.zip'
    print('Downloading FFmpeg from gyan.dev...')
    urllib.request.urlretrieve('https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip', archive)
    with zipfile.ZipFile(archive) as zipped:
        filename = next(n for n in zipped.namelist() if n.endswith('/bin/ffmpeg.exe'))
        with zipped.open(filename) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
    archive.unlink()
print('FFmpeg ready:', target)
