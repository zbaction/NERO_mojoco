#!/usr/bin/env python3
"""把过程图裁到篮子区域放大，便于肉眼核对。"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"


def crop(name, box, scale=2):
    p = OUT / name
    im = Image.open(p).convert("RGB")
    x0, y0, x1, y1 = box
    c = im.crop((x0, y0, x1, y1))
    c = c.resize((c.width * scale, c.height * scale), Image.LANCZOS)
    q = OUT / ("zoom_" + name)
    c.save(q)
    print("写出 %s  (%dx%d)" % (q.name, c.width, c.height))


def bbox_of(color_fn, name):
    a = np.asarray(Image.open(OUT / name).convert("RGB")).astype(np.float32) / 255.0
    mask = color_fn(a)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        print("%s: 无匹配像素" % name)
        return None
    print("%s: 匹配 %d px, bbox x[%d,%d] y[%d,%d]"
          % (name, len(xs), xs.min(), xs.max(), ys.min(), ys.max()))
    return xs.min(), ys.min(), xs.max(), ys.max()


GREEN = lambda a: (a[..., 1] > 0.45) & (a[..., 1] - a[..., 0] > 0.18) & (a[..., 1] - a[..., 2] > 0.18)

for nm in ["pp_1_grasped.png", "pp_5_placed.png", "pp_6_done.png"]:
    b = bbox_of(GREEN, nm)
    if b:
        pad = 90
        crop(nm, (max(0, b[0] - pad), max(0, b[1] - pad),
                  min(1280, b[2] + pad), min(720, b[3] + pad)))
    else:
        crop(nm, (380, 180, 900, 620), 2)
