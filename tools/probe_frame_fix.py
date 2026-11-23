#!/usr/bin/env python3
"""验证 view_scene.frame_camera 是否把相机正确对准机械臂。"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
from PIL import Image

import mujoco
import mujoco.viewer

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out" / "probe"
OUT.mkdir(parents=True, exist_ok=True)

import view_scene as vs  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
d.qpos[:] = m.key("ready").qpos
mujoco.mj_forward(m, d)


def inside_frac(cam):
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        R = d.geom_xmat[g].reshape(3, 3)
        pts.append(m.mesh_vert[a:a + n] @ R.T + d.geom_xpos[g])
    P = np.vstack(pts)
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, np.array([0.0, 0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pos = np.asarray(cam.lookat, float) - fwd * cam.distance
    vv = P - pos
    zc = vv @ fwd
    t = np.tan(np.radians(m.vis.global_.fovy) / 2)
    u = (vv @ right) / (zc * t * 1188 / 704)
    w = (vv @ up) / (zc * t)
    return float(((np.abs(u) < 1) & (np.abs(w) < 1)).mean())


with mujoco.viewer.launch_passive(m, d) as v:
    v.sync()
    time.sleep(0.5)
    print("修复前 lookat=%s dist=%.2f -> 画面内 %.1f%%"
          % (np.round(v.cam.lookat, 3), v.cam.distance, 100 * inside_frac(v.cam)))

    c, span = vs.frame_camera(v, m, d)
    time.sleep(0.5)
    print("修复后 lookat=%s dist=%.2f -> 画面内 %.1f%%"
          % (np.round(v.cam.lookat, 3), v.cam.distance, 100 * inside_frac(v.cam)))

    r = mujoco.Renderer(m, 704, 1188)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = v.cam.lookat
    cam.distance = v.cam.distance
    cam.azimuth = v.cam.azimuth
    cam.elevation = v.cam.elevation
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(OUT / "framed_cam.png")
    r.close()
    print("写出 framed_cam.png")
    for _ in range(100):
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.01)
print("done")
os._exit(0)
