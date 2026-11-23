#!/usr/bin/env python3
"""抓取特写：把镜头怼到夹爪+方块上，确认真的夹住了。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

import mujoco

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_showcase import ARM, SCENE, grab_and_lift  # noqa: E402

OUT = HERE / "out"
W, H = 1280, 720


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in ARM]
    grab_and_lift(m, d, qad)

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
    c = d.xpos[gid].copy()
    print("绿方块中心 %s" % np.round(c, 4))

    r = mujoco.Renderer(m, H, W)
    for name, az, el, dist in [
        ("iso", 42, -16, 0.32),
        ("front", 4, -6, 0.30),
        ("side", 96, -10, 0.30),
    ]:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = c + np.array([0.0, 0.0, 0.004])
        cam.distance = dist
        cam.azimuth = az
        cam.elevation = el
        m.vis.global_.fovy = 34
        r.update_scene(d, cam)
        Image.fromarray(r.render()).save(OUT / ("detail_%s.png" % name))
        print("  写出 detail_%s.png" % name)
    r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
