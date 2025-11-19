"""(Deprecated) Automated sample download script.

Public hosting links used earlier became unreliable (404 / 403). For reliability
please manually download 1-2 retina images and place them as:

data/retina/healthy/sample1.jpg
data/retina/risk/sample2.jpg

Then run training. This script kept only as a placeholder.
"""
import os
import urllib.request
from urllib.error import URLError, HTTPError

SAMPLES = [
    # Fallback small retina-like images (generic) - purely to exercise pipeline.
    ("https://raw.githubusercontent.com/dylanede/cyclegan/master/datasets/monet2photo/trainA/00001.jpg", "healthy", "retina_healthy_demo.jpg"),
    ("https://raw.githubusercontent.com/dylanede/cyclegan/master/datasets/monet2photo/trainB/00001.jpg", "risk", "retina_risk_demo.jpg"),
]


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def download():
    base = os.path.join("data", "retina")
    ensure_dir(base)
    print("This script is deprecated. Please manually add images.")


if __name__ == "__main__":
    download()
