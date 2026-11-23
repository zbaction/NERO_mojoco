#!/usr/bin/env python3
"""检查渲染图里三个方块的颜色是否确实是三种不同颜色（不靠肉眼）。"""
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent


def analyse(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    vivid = (sat > 0.45) & (mx > 0.25)

    masks = {
        "red":   vivid & (r - g > 0.35) & (r - b > 0.35),
        "green": vivid & (g - r > 0.20) & (g - b > 0.12),
        "blue":  vivid & (b - r > 0.20) & (b - g > 0.12),
    }
    total = img.shape[0] * img.shape[1]
    means, counts = {}, {}
    for name, m in masks.items():
        px = int(m.sum())
        counts[name] = px
        means[name] = img[m].mean(axis=0) if px else np.zeros(3)
        print("  {:>5}: {:>6} px ({:5.2f}%)  mean_rgb={}".format(
            name, px, 100.0 * px / total, np.round(means[name], 3)))

    ok = True
    for name, px in counts.items():
        if px < 600:
            print("  ! {} 像素太少 ({})".format(name, px))
            ok = False
    names = list(means)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm(means[names[i]] - means[names[j]]))
            print("  |{}-{}| = {:.3f}".format(names[i], names[j], d))
            if d < 0.25:
                print("  ! 两种颜色过于接近")
                ok = False
    return ok


def main():
    files = [pathlib.Path(p) for p in sys.argv[1:]] or sorted((HERE / "out").glob("desk_*.png"))
    if not files:
        print("没有 out/desk_*.png，先运行 view.py --render")
        return 1
    all_ok = True
    for f in files:
        print("[{}]".format(f.name))
        all_ok = analyse(f) and all_ok
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())