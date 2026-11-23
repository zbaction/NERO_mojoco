#!/usr/bin/env python3
"""检查过程图：非空、有绿方块像素、有木色篮子像素。"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"
for p in sorted(OUT.glob("pp_*.png")):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = a.shape
    # 绿方块：G 明显高于 R、B
    green = (a[..., 1] > 0.45) & (a[..., 1] - a[..., 0] > 0.18) & (a[..., 1] - a[..., 2] > 0.18)
    # 木色篮子：R>G>B 且整体偏亮中等
    wood = (a[..., 0] > 0.30) & (a[..., 0] - a[..., 2] > 0.12) & (a[..., 1] - a[..., 2] > 0.05) & (a[..., 0] < 0.85)
    uniq = len(np.unique((a * 63).astype(np.uint8).reshape(-1, 3), axis=0))
    print("%-24s %dx%d  mean %.3f  std %.3f  量化色 %5d  绿 %5d px  木 %6d px"
          % (p.name, w, h, a.mean(), a.std(), uniq, green.sum(), wood.sum()))
