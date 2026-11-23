#!/usr/bin/env python3
"""把成果图裁切放大，方便检查细节。"""
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"


def crop(src, box, dst, scale=2):
    im = Image.open(OUT / src).convert("RGB").crop(box)
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    im.save(OUT / dst)
    a = np.asarray(im).astype(float)
    lum = a @ [0.2126, 0.7152, 0.0722]
    print("%-22s %s mean=%.1f p95=%.1f" % (dst, a.shape, lum.mean(), np.percentile(lum, 95)))


# showcase_front：机械臂大致在画面中部偏左
crop("showcase_front.png", (420, 60, 900, 660), "z_front_crop.png", 2)
crop("showcase_grasp.png", (380, 100, 860, 620), "z_grasp_crop.png", 2)
crop("showcase_close.png", (300, 120, 980, 700), "z_close_crop.png", 2)
crop("showcase_side.png", (380, 80, 900, 640), "z_side_crop.png", 2)
