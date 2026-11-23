#!/usr/bin/env python3
"""量夹爪真实开度：打印 pad 完整坐标 + 两 pad 相对位姿。"""
import pathlib

import numpy as np
import mujoco

SCENE = pathlib.Path.home() / "NERO" / "nero_table_scene.xml"
m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)

pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
pr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_l")
jr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_r")
print("pad 尺寸 L=%s R=%s" % (m.geom_size[pl], m.geom_size[pr]))
print("pad body L=%d R=%d" % (m.geom_bodyid[pl], m.geom_bodyid[pr]))
print("body 名 L=%s R=%s"
      % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[pl]),
         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[pr])))

for q in (0.0, 0.050):
    d.qpos[m.jnt_qposadr[jl]] = q
    d.qpos[m.jnt_qposadr[jr]] = q
    mujoco.mj_forward(m, d)
    print("qpos=%.3f  L=%s  R=%s  |Δ|=%.4f m"
          % (q, np.round(d.geom_xpos[pl], 4), np.round(d.geom_xpos[pr], 4),
             np.linalg.norm(d.geom_xpos[pl] - d.geom_xpos[pr])))
