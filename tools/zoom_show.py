#!/usr/bin/env python3
"""放大检查展示图：夹爪是否真的夹住了绿块、机械臂细节是否清楚。"""
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"


def zoom(src, box, dst, scale=2):
    im = Image.open(OUT / src).convert("RGB").crop(box)
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    im.save(OUT / dst)
    a = np.asarray(im).astype(float)
    lum = a @ [0.2126, 0.7152, 0.0722]
    # 绿色像素（夹爪抓的绿块）
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > 90) & (g > r * 1.45) & (g > b * 1.45)
    print("%-20s %s lum mean=%.0f std=%.0f p2=%.0f p98=%.0f  绿像素=%d"
          % (dst, a.shape, lum.mean(), lum.std(),
             np.percentile(lum, 2), np.percentile(lum, 98), green.sum()))


zoom("showcase_grasp.png", (300, 300, 1000, 720), "z_zoom_grasp.png", 2)
zoom("showcase_grasp.png", (430, 380, 780, 720), "z_zoom_grip.png", 3)
zoom("showcase_iso.png", (330, 330, 1030, 720), "z_zoom_iso.png", 2)
