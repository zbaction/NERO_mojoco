#!/usr/bin/env python3
"""诊断：走到 grasp 位姿后为什么指尖没到位 —— 打印关节跟踪误差 + 所有接触对。"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
QAD = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
EE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")


def pad_mid():
    return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])


def contacts_with_arm(limit=25):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        nm1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "?"
        nm2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "?"
        arm1 = nm1 in ("nero_base", "link1", "link2", "link3", "link4", "link5",
                       "link6", "link7", "end_effector", "gripper_base",
                       "finger_l", "finger_r")
        arm2 = nm2 in ("nero_base", "link1", "link2", "link3", "link4", "link5",
                       "link6", "link7", "end_effector", "gripper_base",
                       "finger_l", "finger_r")
        if arm1 or arm2:
            out.append("%-18s <-> %-18s dist=%+.5f" % (n1, n2, c.dist))
    return out[:limit]


d.qpos[:] = m.key("ready").qpos
d.ctrl[:7] = [m.key("ready").qpos[a] for a in QAD]
d.ctrl[7] = d.ctrl[8] = 0.05
mujoco.mj_forward(m, d)

for name, secs in (("hover", 2.0), ("grasp", 3.0)):
    tgt = np.array([m.key(name).qpos[a] for a in QAD])
    d.ctrl[:7] = tgt
    d.ctrl[7] = d.ctrl[8] = 0.05
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)
    cur = np.array([d.qpos[a] for a in QAD])
    print("== %s ==" % name)
    print("  跟踪误差(度): " + " ".join("%+.2f" % x for x in np.degrees(cur - tgt)))
    print("  指尖中点 [%.3f %.3f %.3f]  目标 [%.3f %.3f %.3f]"
          % (*pad_mid(), *[0.180, 0.0, 0.790 if name == "grasp" else 0.890]))
    cs = contacts_with_arm()
    print("  机械臂相关接触 %d 个:" % len(cs))
    for c in cs:
        print("     " + c)

# 抬起来看看
print("\n== 闭合 + 抬起 ==")
for _ in range(int(2.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
print("  闭合后指尖中点 [%.3f %.3f %.3f]" % tuple(pad_mid()))
green = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
print("  green [%.4f %.4f %.4f]" % tuple(d.xpos[green]))
for c in contacts_with_arm():
    print("     " + c)
