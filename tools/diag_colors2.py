#!/usr/bin/env python3
"""诊断 red/blue 走 hover→grasp 时的接触和关节跟踪误差。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402

m, d, qad = pp.load()
JOINTS = ["joint%d" % i for i in range(1, 8)]


def contacts(tag):
    rows = []
    for k in range(d.ncon):
        c = d.contact[k]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        rows.append("%s<->%s %.4f" % (g1, g2, c.dist))
    print("    %s: %d 个接触" % (tag, len(rows)))
    for r in rows:
        print("      " + r)


for col in ("red", "green", "blue"):
    print("=== %s ===" % col)
    pp.reset(m, d, qad)
    pp.move(m, d, qad, "%s_hover" % col, pp.GRIP_OPEN, 2.0)
    q_hover = np.array([d.qpos[a] for a in qad])
    contacts("hover 后")

    # 逐帧记录关节跟踪误差
    q_cmd = np.array([m.key("%s_grasp" % col).qpos[a] for a in qad])
    d.ctrl[:7] = q_cmd
    d.ctrl[7] = d.ctrl[8] = pp.GRIP_OPEN
    worst = 0.0
    for i in range(int(2.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        err = np.abs(d.qpos[qad] - q_cmd)
        if err.max() > worst:
            worst = err.max()
    q_act = np.array([d.qpos[a] for a in qad])
    print("    关节误差(度): " + " ".join("%+.1f" % np.degrees(v)
                                         for v in (q_act - q_cmd)))
    print("    最大关节误差 %.2f 度" % np.degrees(worst))
    contacts("grasp 后")
    print()
