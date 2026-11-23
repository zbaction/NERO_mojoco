#!/usr/bin/env python3
"""`--all` 跑完后俯视出图，确认三块方块都还在篮子里。"""
from __future__ import annotations

import math
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402
from PIL import Image  # noqa: E402

OUT = HERE / "out"


def main():
    m, d, qad = pp.load()
    for col in pp.COLORS:
        ok = pp.run(m, d, qad, col, verbose=False)
        print("%-5s -> %s" % (col, "PASS" if ok else "FAIL"))
    m.vis.global_.fovy = 40
    r = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    # 俯视篮子
    (bx, by), ft, inner = pp.basket_box(m, d)
    cam.lookat[:] = [bx, by, ft + 0.05]
    cam.distance = 0.95
    cam.azimuth = 90
    cam.elevation = -68
    r.update_scene(d, cam)
    img = r.render()
    Image.fromarray(img).save(OUT / "basket_three_cubes.png")
    r.close()
    print("写出 out/basket_three_cubes.png")
    for col in pp.COLORS:
        ok, p, sc, ft, inner = pp.cube_in_basket(m, d, pp.CUBE_OF[col])
        print("  %-5s [%.3f %.3f %.3f] 偏离格位 %.1f mm -> %s"
              % (col, *p, 1000 * math.hypot(p[0] - sc[0], p[1] - sc[1]),
                 "PASS" if ok else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
