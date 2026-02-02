import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
file_task = "face_landmarker.task"

if not os.path.exists(file_task):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, file_task)
        print("Download complete.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Model file already exists.")
