#!/usr/bin/env python3
"""按色相分簇统计出图里的红/绿/蓝方块像素（排除木质桌面）。"""
from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 pathlib.Path.home() / "NERO/out/basket_three_cubes.png")
a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255.0
h, w, _ = a.shape
mx = a.max(axis=2)
mn = a.min(axis=2)
sat = (mx - mn) / np.maximum(mx, 1e-9)
# 向量化 hue
r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
d = mx - mn + 1e-12
hue = np.zeros_like(mx)
m = (mx == r)
hue[m] = ((g[m] - b[m]) / d[m]) % 6
m = (mx == g) & (mx != r)
hue[m] = (b[m] - r[m]) / d[m] + 2
m = (mx == b) & (mx != r) & (mx != g)
hue[m] = (r[m] - g[m]) / d[m] + 4
hue *= 60.0

print("尺寸 %dx%d  亮度均值 %.1f  唯一色 %d"
      % (w, h, a.mean() * 255, len(np.unique((a * 255).astype(np.uint8)
                                            .reshape(-1, 3), axis=0))))

# 只统计饱和度高、亮度够的像素，再按色相分箱
strong = (sat > 0.55) & (mx > 0.28)

for name, lo, hi in (("red", -1, 20), ("red2", 340, 361),
                     ("green", 90, 165), ("blue", 195, 265)):
    if lo < 0:
        mask = strong & ((hue < hi) | (hue >= 360 + lo))
    else:
        mask = strong & (hue >= lo) & (hue < hi)
    n = int(mask.sum())
    if n < 50:
        print("  %-6s %6d px" % (name, n))
        continue
    mean = a[mask].mean(axis=0) * 255
    ys, xs = np.where(mask)
    print("  %-6s %6d px  均值 rgb [%.0f %.0f %.0f]  bbox x[%d,%d] y[%d,%d]"
          % (name, n, *mean, xs.min(), xs.max(), ys.min(), ys.max()))

# 木质桌面/篮子大概什么色相？取面积最大的 hue 峰
hist, edges = np.histogram(hue[strong], bins=36, range=(0, 360))
order = np.argsort(hist)[::-1][:5]
print("\n最饱和区色相峰（占比）：")
tot = max(int(strong.sum()), 1)
for i in order:
    print("  %3d–%3d°  %7d px  %.1f%%"
          % (edges[i], edges[i + 1], hist[i], 100.0 * hist[i] / tot))
