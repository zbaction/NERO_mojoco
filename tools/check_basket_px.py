#!/usr/bin/env python3
"""用「隐藏 basket body 前后两次渲染差分」客观测篮子在各机位的可见像素。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import mujoco
from PIL import Image

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))


def render(m, d, cam, w=1280, h=720):
    r = mujoco.Renderer(m, h, w)
    m.vis.global_.fovy = 42
    r.update_scene(d, cam)
    img = r.render().copy()
    r.close()
    return img


def main():
    m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "basket")
    # basket 及其所有 geom
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    print("篮子 body id=%d, geom %d 个: %s"
          % (bid, len(geoms),
             [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms]))

    cams = {"overview": ("overview", None), "top": ("top", None)}
    for nm, (xml_cam, _) in cams.items():
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, xml_cam)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cid
        a = render(m, d, cam)

        # 隐藏 basket 的所有 geom
        saved = [m.geom_rgba[g].copy() for g in geoms]
        saved_vis = [m.geom_group[g] for g in geoms]
        for g in geoms:
            m.geom_group[g] = 3   # 只关掉 geomgroup 0/1/2 时不可见
        b = render(m, d, cam)
        for g, c, v in zip(geoms, saved, saved_vis):
            m.geom_rgba[g] = c
            m.geom_group[g] = v

        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        n = int((diff > 8).sum())
        frac = 100.0 * n / (a.shape[0] * a.shape[1])
        print("%-10s 篮子像素 %6d  %.2f%%" % (xml_cam, n, frac))
        if n:
            ys, xs = np.nonzero(diff > 8)
            print("           bbox x[%d,%d] y[%d,%d]" % (xs.min(), xs.max(), ys.min(), ys.max()))
            Image.fromarray(a).save(HERE / "out" / ("basket_%s.png" % xml_cam))
            Image.fromarray(b).save(HERE / "out" / ("nobasket_%s.png" % xml_cam))


if __name__ == "__main__":
    main()
