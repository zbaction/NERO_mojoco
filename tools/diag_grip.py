#!/usr/bin/env python3
"""诊断夹爪闭合：手指到底动没动？pad 与方块的真实相对位置是多少？"""
from __future__ import annotations

import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
QAD = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
GL = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_l")]
GR = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_r")]
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
GREEN = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_green_geom")
print("夹爪执行器:")
for i in range(m.nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print("  ctrl[%d] = %-12s trnid=%s gainprm0=%.0f ctrlrange=%s"
          % (i, nm, m.actuator_trnid[i], m.actuator_gainprm[i][0], m.actuator_ctrlrange[i]))
print("grip 关节 range: L=%s R=%s" % (m.jnt_range[m.jnt_qposadr[GL] - 0], m.jnt_range[GR]))
print("pad geom size: L=%s R=%s" % (m.geom_size[PADL], m.geom_size[PADR]))

d.qpos[:] = m.key("grasp").qpos
d.qpos[GL] = d.qpos[GR] = 0.045
for i, a in enumerate(QAD):
    d.ctrl[i] = d.qpos[a]
d.ctrl[7] = d.ctrl[8] = 0.045
mujoco.mj_forward(m, d)
print("\n张开(0.045): padL=%s padR=%s 内间隙=%.1f mm"
      % (np.round(d.geom_xpos[PADL], 4), np.round(d.geom_xpos[PADR], 4),
         1000 * (d.geom_xpos[PADL][1] - d.geom_xpos[PADR][1] - 2 * 0.006)))
print("        green center %s" % np.round(d.xpos[mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")], 4))

d.ctrl[7] = d.ctrl[8] = 0.0
for k in range(0, 2001):
    mujoco.mj_step(m, d)
    if k % 250 == 0:
        print("  t=%.2fs  gripL=%.4f gripR=%.4f  padL_y=%+.4f padR_y=%+.4f  ncon=%d"
              % (k * m.opt.timestep, d.qpos[GL], d.qpos[GR],
                 d.geom_xpos[PADL][1], d.geom_xpos[PADR][1], d.ncon))
print("\n最终 gripL=%.4f gripR=%.4f" % (d.qpos[GL], d.qpos[GR]))
print("padL=%s padR=%s" % (np.round(d.geom_xpos[PADL], 4), np.round(d.geom_xpos[PADR], 4)))
print("green=%s" % np.round(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                                     "cube_green")], 4))
print("所有接触:")
for i in range(d.ncon):
    c = d.contact[i]
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
    print("   %-20s <-> %-20s dist=%+.6f" % (n1, n2, c.dist))

# 几何上 pad 能不能碰到方块：直接算包围盒距离
print("\npad 与绿方块的几何关系:")
for nm, g in (("padL", PADL), ("padR", PADR)):
    print("  %s xpos=%s size=%s" % (nm, np.round(m.geom_xpos[g], 4), m.geom_size[g]))
print("  green xpos=%s size=%s" % (np.round(m.geom_xpos[GREEN], 4), m.geom_size[GREEN]))
for nm, g in (("padL", PADL), ("padR", PADR)):
    dist = np.abs(m.geom_xpos[g] - m.geom_xpos[GREEN]) - (m.geom_size[g] + m.geom_size[GREEN])
    print("  %s 到方块的面间隙 abc=%s" % (nm, np.round(dist, 4)))
