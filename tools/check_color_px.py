#!/usr/bin/env python3
"""统计三色过程图里方块像素数，确认每种颜色确实出现在画面里。"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"
RGB = {"red": (0.86, 0.13, 0.13), "green": (0.12, 0.68, 0.26),
       "blue": (0.16, 0.34, 0.86)}

for col, ref in RGB.items():
    ref = np.array(ref, float)
    print("=== %s (参考色 %s) ===" % (col, np.round(ref, 2)))
    for n in range(1, 8):
        cands = sorted(OUT.glob("pp_%s_%d_*.png" % (col, n)))
        if not cands:
            continue
        p = cands[0]
        a = np.asarray(Image.open(p).convert("RGB"), float) / 255.0
        sat = a.max(2) - a.min(2)
        dist = np.linalg.norm(a - ref[None, None, :], axis=2)
        mask = (dist < 0.35) & (sat > 0.25)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            print("  %-28s 无匹配像素" % p.name)
            continue
        print("  %-28s %5d px  bbox x[%4d,%4d] y[%4d,%4d]"
              % (p.name, len(xs), xs.min(), xs.max(), ys.min(), ys.max()))
