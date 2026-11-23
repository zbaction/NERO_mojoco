#!/usr/bin/env python3
"""定位 ctrl 的单步跳变：哪一步、哪个关节、跳了多大、在哪条轨迹里。"""
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
    tag = []

    rec_c, rec_tag = [], []
    orig = mujoco.mj_step

    def hook(mm, dd):
        orig(mm, dd)
        rec_c.append(np.array(dd.ctrl[:7]))
        rec_tag.append(tag[0] if tag else "")

    mujoco.mj_step = hook
    try:
        for item in pp.plan_for(col):
            kind, keys, grip = item[0], item[1], item[2]
            hold = item[4] if len(item) > 4 else ()
            slow = item[5] if len(item) > 5 else ()
            tag.clear()
            tag.append("%s %s" % (kind, "→".join(str(k) for k in keys)))
            if kind == "path":
                pp.move_path(m, d, qad, keys, grip, None, hold=hold, slow=slow)
            else:
                pp.dwell(m, d, qad, grip)
    finally:
        mujoco.mj_step = orig

    C = np.array(rec_c)
    dc = np.abs(np.diff(C, axis=0))
    dd = dc.max(axis=1) / dt
    order = np.argsort(dd)[::-1]
    print("ctrl 单步跳变最大的 12 处：")
    for i in order[:12]:
        j = int(np.argmax(dc[i]))
        print("  t=%6.3f s  关节 j%d  Δ=%.4f rad = %.2f°  → %.1f rad/s（%.0f °/s）  [%s]"
              % ((i + 1) * dt, j + 1, dc[i, j], np.degrees(dc[i, j]), dd[i], np.degrees(dd[i]),
                 rec_tag[i]))
    print()
    print("跳变 > 0.05 rad 的总共 %d 处" % int((dc > 0.05).any(axis=1).sum()))
    idx = np.where((dc > 0.05).any(axis=1))[0]
    for i in idx[:20]:
        j = int(np.argmax(dc[i]))
        print("  t=%6.3f  j%d  Δ=%.3f rad  [%s]" % ((i + 1) * dt, j + 1, dc[i, j], rec_tag[i]))


if __name__ == "__main__":
    main()
