"""Build demo/field_walk_SYNTHETIC.mp4: a slow pan-and-zoom over six LeafNet75 test photos, standing in for a
scouting video. SYNTHETIC: no openly licensed, annotated field video of these diseases exists (docs/DATASET_CARD.md)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ["00018", "00026", "00058", "00089", "00094", "00127"]
W, H, FPS, SECONDS = 960, 540, 15, 3


def main():
    out = cv2.VideoWriter(str(ROOT / "demo" / "field_walk_SYNTHETIC.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k, name in enumerate(PHOTOS):
        img = cv2.imread(str(ROOT / "data" / "leafseg" / "images" / f"{name}.jpg"))
        scale = max(W / img.shape[1], H / img.shape[0]) * 1.25
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        n = FPS * SECONDS
        for f in range(n):
            t = f / (n - 1)
            zoom = 1 + 0.15 * t  # slow push-in while panning, like walking along a row
            cw, ch = int(W / zoom * 1.0), int(H / zoom * 1.0)
            x = int((img.shape[1] - cw) * (t if k % 2 == 0 else 1 - t))
            y = int((img.shape[0] - ch) / 2)
            frame = cv2.resize(img[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA)
            out.write(frame)
    out.release()


if __name__ == "__main__":
    main()
