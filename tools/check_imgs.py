#!/usr/bin/env python3
"""检查渲染图的亮度分布，判断是不是太暗/构图太空。"""
import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path.home() / "NERO" / "out"
for p in sorted(OUT.glob("*.png")):
    a = np.asarray(Image.open(p).convert("RGB")).astype(float)
    lum = a @ [0.2126, 0.7152, 0.0722]
    print("%-24s %s  亮度 mean=%5.1f p50=%5.1f p95=%5.1f max=%5.1f  暗(<40)=%4.1f%%"
          % (p.name, a.shape, lum.mean(), np.percentile(lum, 50),
             np.percentile(lum, 95), lum.max(), 100 * (lum < 40).mean()))
