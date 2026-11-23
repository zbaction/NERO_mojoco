#!/usr/bin/env python3
"""诊断抬起：夹住之后为什么方块不跟着走。"""
from __future__ import annotations

import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
QAD = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
GREEN = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
GEOM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_green_geom")


def padmid():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


def report(tag):
    con = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == GEOM or c.geom2 == GEOM:
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
            if "finger" in n1 + n2:
                con.append("%s|%s %.5f" % (n1, n2, c.dist))
    print("  %-8s padmid=%s green=%s ncon_grip=%d %s"
          % (tag, np.round(padmid(), 4), np.round(d.xpos[GREEN], 4),
             len(con), ";".join(con)))


d.qpos[:] = m.key("ready").qpos
d.ctrl[:7] = [m.key("ready").qpos[a] for a in QAD]
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)


def drive(qtgt, grip, secs, tag=""):
    d.ctrl[:7] = qtgt
    d.ctrl[7] = d.ctrl[8] = grip
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if tag:
        report(tag)


q_hover = np.array([m.key("hover").qpos[a] for a in QAD])
q_grasp = np.array([m.key("grasp").qpos[a] for a in QAD])
drive(q_hover, 0.05, 2.0, "hover")
drive(q_grasp, 0.05, 2.5, "grasp")
drive(q_grasp, 0.0, 2.0, "close")
print("  指关节 qpos: L=%.4f R=%.4f" % (d.qpos[QAD[-1] + 1], d.qpos[QAD[-1] + 2]))

print("\n-- 方案 A：joint2 抬 -20° --")
d.qpos[:] = m.key("grasp").qpos  # 重新夹住
d.ctrl[:7] = q_grasp
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)
drive(q_hover, 0.05, 2.0)
drive(q_grasp, 0.05, 2.5)
drive(q_grasp, 0.0, 2.0, "closed")
za = d.xpos[GREEN][2]
ql = q_grasp.copy()
ql[1] += np.radians(-20)
for k in range(int(2.0 / m.opt.timestep)):
    d.ctrl[:7] = ql
    mujoco.mj_step(m, d)
    if k % 250 == 0:
        report("t=%.1f" % (k * m.opt.timestep))
report("final")
print("  A: dz=%+.4f" % (d.xpos[GREEN][2] - za))

print("\n-- 方案 B：直接竖直回到 hover --")
d.qpos[:] = m.key("grasp").qpos
d.ctrl[:7] = q_grasp
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)
drive(q_hover, 0.05, 2.0)
drive(q_grasp, 0.05, 2.5)
drive(q_grasp, 0.0, 2.0, "closed")
zb = d.xpos[GREEN][2]
for k in range(int(3.0 / m.opt.timestep)):
    d.ctrl[:7] = q_hover
    mujoco.mj_step(m, d)
    if k % 400 == 0:
        report("t=%.1f" % (k * m.opt.timestep))
report("final")
print("  B: dz=%+.4f" % (d.xpos[GREEN][2] - zb))

print("\n-- 方案 C：夹紧力加大（ctrl=-0.02 过盈）--")
d.qpos[:] = m.key("grasp").qpos
d.ctrl[:7] = q_grasp
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)
drive(q_hover, 0.05, 2.0)
drive(q_grasp, 0.05, 2.5)
drive(q_grasp, -0.02, 2.0, "closed")
zc = d.xpos[GREEN][2]
for k in range(int(3.0 / m.opt.timestep)):
    d.ctrl[:7] = q_hover
    d.ctrl[7] = d.ctrl[8] = -0.02
    mujoco.mj_step(m, d)
    if k % 400 == 0:
        report("t=%.1f" % (k * m.opt.timestep))
report("final")
print("  C: dz=%+.4f" % (d.xpos[GREEN][2] - zc))
