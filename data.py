import os
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

BASE = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/"
CACHE = "data/xray_128.npz"
IMG_SIZE = 128


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def load_image(name):
    try:
        raw = fetch(BASE + "images/" + name)
        img = Image.open(io.BytesIO(raw)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        print("skip", name, e)
        return None


def build_cache():
    os.makedirs("data", exist_ok=True)
    meta = pd.read_csv(io.BytesIO(fetch(BASE + "metadata.csv")))
    # keep frontal chest xrays only, drop CT scans and side views
    meta = meta[(meta["modality"] == "X-ray") & (meta["view"].isin(["PA", "AP"]))]
    meta = meta[meta["finding"] != "todo"]
    meta = meta.reset_index(drop=True)
    labels = (meta["finding"] == "Pneumonia/Viral/COVID-19").astype(int).values
    names = meta["filename"].tolist()
    print("downloading", len(names), "images")

    with ThreadPoolExecutor(max_workers=16) as ex:
        imgs = list(ex.map(load_image, names))

    keep = [i for i, im in enumerate(imgs) if im is not None]
    X = np.stack([imgs[i] for i in keep])
    y = labels[keep]
    patient = meta["patientid"].values[keep]
    np.savez_compressed(CACHE, X=X, y=y, patient=patient)
    print("saved", X.shape, "positives", int(y.sum()))


def load():
    if not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=True)
    return d["X"], d["y"], d["patient"]


if __name__ == "__main__":
    build_cache()
