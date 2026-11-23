#!/usr/bin/env python3
"""跟踪误差：指令轨迹 vs 实际关节角，判断"曲线本身平滑但伺服跟不上"。"""
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
    rc, rq, tag, tabs = [], [], [""], []
    orig = mujoco.mj_step

    def hook(mm, dd):
        orig(mm, dd)
        rc.append(np.array(dd.ctrl[:7]))
        rq.append(np.array(dd.qpos[qad]))
        tabs.append(tag[0])

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

    C = np.degrees(np.array(rc))
    Q = np.degrees(np.array(rq))
    err = np.abs(C - Q)
    print("跟踪误差（指令 - 实际）：整体峰值 %.2f°   RMS %.2f°"
          % (err.max(), np.sqrt((err ** 2).mean())))
    print()
    for name in sorted(set(tabs)):
        if not name.strip():
            continue
        mask = np.array([t == name for t in tabs])
        if mask.sum() == 0:
            continue
        e = err[mask]
        print("  %-52s 峰值 %5.2f°  RMS %5.2f°" % (name, e.max(), np.sqrt((e ** 2).mean())))
    print()
    i = int(np.argmax(err.max(axis=1)))
    j = int(np.argmax(err[i]))
    print("峰值出现在 t=%.3f s 关节 j%d：指令 %.2f° 实际 %.2f°  [%s]"
          % ((i + 1) * dt, j + 1, C[i, j], Q[i, j], tabs[i]))


if __name__ == "__main__":
    main()
