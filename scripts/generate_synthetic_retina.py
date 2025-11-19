"""Generate synthetic retinal-like images for demo purposes.
This is NOT medical data. It only enables the image pipeline to run.
"""
import os, math, random
from PIL import Image, ImageDraw
from typing import Tuple

OUTPUT_BASE = "data/retina"

SETTINGS = {
    "healthy": {
        "vessels": (10, 15),
        "bleeds": (0, 0),
        "disc_color": (30, 80, 30),
        "vessel_color": (0, 140, 0)
    },
    "risk": {
        "vessels": (18, 28),
        "bleeds": (25, 45),
        "disc_color": (55, 40, 40),
        "vessel_color": (180, 0, 0)
    }
}

def ensure_dirs():
    for cls in SETTINGS.keys():
        os.makedirs(os.path.join(OUTPUT_BASE, cls), exist_ok=True)


def rand_in(a: Tuple[int, int]):
    return random.randint(a[0], a[1])


def draw_retina(path: str, cls: str, seed: int):
    cfg = SETTINGS[cls]
    random.seed(seed)
    img = Image.new("RGB", (224, 224), "black")
    d = ImageDraw.Draw(img)
    # main disc
    d.ellipse((12, 12, 212, 212), fill=cfg["disc_color"])
    cx, cy = 112, 112
    for _ in range(rand_in(cfg["vessels"])):
        angle = random.random() * 2 * math.pi
        length = random.randint(60, 105)
        x2 = cx + int(math.cos(angle) * length)
        y2 = cy + int(math.sin(angle) * length)
        width = 2 if cls == "healthy" else random.randint(2, 5)
        d.line((cx, cy, x2, y2), fill=cfg["vessel_color"], width=width)
    for _ in range(rand_in(cfg["bleeds"])):
        x = random.randint(40, 184)
        y = random.randint(40, 184)
        r = random.randint(2, 5)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(220, 20, 20))
    img.save(path)


def generate_per_class(count: int = 20):
    ensure_dirs()
    for cls in SETTINGS.keys():
        for i in range(count):
            path = os.path.join(OUTPUT_BASE, cls, f"{cls}_{i}.png")
            draw_retina(path, cls, seed=1000 * (1 if cls == "risk" else 0) + i)
    print(f"Generated synthetic images under {OUTPUT_BASE}")

if __name__ == "__main__":
    generate_per_class(count=25)
