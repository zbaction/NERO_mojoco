#!/usr/bin/env python3
"""定位指尖加速度/加加速度峰值出现在什么时候、属于哪个动作。"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402


def main():
    col = sys.argv[1] if len(sys.argv) > 1 else "green"
    m, d, qad = pp.load()
    pp.reset(m, d, qad)
    dt = m.opt.timestep
    padl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
    padr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")

    rec_p, rec_tag = [], []
    tag = [""]
    orig = mujoco.mj_step

    def hook(mm, dd):
        orig(mm, dd)
        rec_p.append(0.5 * (dd.geom_xpos[padl] + dd.geom_xpos[padr]).copy())
        rec_tag.append(tag[0])

    mujoco.mj_step = hook
    try:
        for item in pp.plan_for(col):
            kind, keys, grip = item[0], item[1], item[2]
            hold = item[4] if len(item) > 4 else ()
            slow = item[5] if len(item) > 5 else ()
            tag[0] = "%s %s" % (kind, "→".join(str(k) for k in keys))
            if kind == "path":
                pp.move_path(m, d, qad, keys, grip, None, hold=hold, slow=slow)
            else:
                pp.dwell(m, d, qad, grip)
    finally:
        mujoco.mj_step = orig

    P = np.array(rec_p)
    v = np.gradient(P, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    jk = np.gradient(a, dt, axis=0)
    sa = np.linalg.norm(a, axis=1)
    sj = np.linalg.norm(jk, axis=1)
    sp = np.linalg.norm(v, axis=1)

    print("指尖速度峰值 %.3f m/s" % sp.max())
    print()
    print("加速度峰值 top 10：")
    for i in np.argsort(sa)[::-1][:10]:
        print("  t=%6.3f s  |a|=%7.2f m/s²   |v|=%5.3f    [%s]"
              % ((i + 1) * dt, sa[i], sp[i], rec_tag[i]))
    print()
    print("jerk 峰值 top 10：")
    for i in np.argsort(sj)[::-1][:10]:
        print("  t=%6.3f s  |j|=%9.1f m/s³   |a|=%6.2f   [%s]"
              % ((i + 1) * dt, sj[i], sa[i], rec_tag[i]))
    print()
    # 每 0.5 s 打印一次速度，看整体节奏
    print("速度剖面（每 0.5 s）：")
    step = int(0.5 / dt)
    line = []
    for i in range(0, len(P), step):
        line.append("%.2fs:%.3f" % (i * dt, sp[i]))
        if len(line) == 8:
            print("  " + "  ".join(line))
            line = []
    if line:
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
