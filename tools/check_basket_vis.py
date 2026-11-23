#!/usr/bin/env python3
"""检查成果图里能不能看到篮子（木色材质）。"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"
for nm in ["nero_overview.png", "showcase_grasp.png", "showcase_front.png",
           "showcase_side.png", "showcase_iso.png"]:
    p = OUT / nm
    if not p.exists():
        print("%-24s 缺失" % nm)
        continue
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    # 木色 mat_basket rgba (0.55,0.40,0.22) 渲染后大致 R>0.35, R-B>0.15
    wood = (a[..., 0] > 0.34) & (a[..., 0] - a[..., 2] > 0.16) & (a[..., 1] > 0.22) & (a[..., 1] < 0.75)
    ys, xs = np.nonzero(wood)
    if len(xs):
        print("%-24s 木色 %6d px  bbox x[%d,%d] y[%d,%d]"
              % (nm, len(xs), xs.min(), xs.max(), ys.min(), ys.max()))
    else:
        print("%-24s 木色      0 px" % nm)
