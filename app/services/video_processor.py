import tempfile
import requests
import os

def download_video_temp(url: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_path