import urllib.request
import bz2
import os

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
file_bz2 = "shape_predictor_68_face_landmarks.dat.bz2"
file_dat = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(file_dat):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, file_bz2)
        print("Download complete. Extracting...")
        with bz2.BZ2File(file_bz2) as fr, open(file_dat, 'wb') as fw:
            fw.write(fr.read())
        print("Extraction complete.")
        # os.remove(file_bz2) # Keep it just in case, or remove. I'll remove.
        if os.path.exists(file_bz2):
             os.remove(file_bz2)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File already exists.")
