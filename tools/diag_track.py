#!/usr/bin/env python3
"""诊断位置伺服的跟踪质量：跟踪误差、阻尼比、有没有振铃。"""
from __future__ import annotations

import math
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402


def main():
    m, d, qad = pp.load()
    pp.reset(m, d, qad)
    dt = m.opt.timestep

    # 段 5（green_transit）是最长的一段，用它看跟踪
    for key in ("green_transit",):
        q0 = np.array([d.qpos[a] for a in qad], dtype=float)
        q1 = np.array([m.key(key).qpos[a] for a in qad], dtype=float)
        cnt = int(2.5 / dt)
        Q, C = [], []
        for k in range(cnt):
            s = 0.5 * (1.0 - math.cos(math.pi * (k + 1) / cnt))
            d.ctrl[:7] = q0 + (q1 - q0) * s
            mujoco.mj_step(m, d)
            Q.append(np.array(d.qpos[qad]))
            C.append(np.array(d.ctrl[:7]))
        Q, C = np.array(Q), np.array(C)
        E = C - Q
        print("段 %s：cmd 总位移 %.1f°" % (key, np.degrees(np.abs(q1 - q0).max())))
        print("  跟踪误差 峰值 %.2f°   RMS %.2f°  末值 %.2f°"
              % (np.degrees(np.abs(E).max()),
                 np.degrees(np.sqrt((E ** 2).mean())),
                 np.degrees(np.abs(E[-1]).max())))
        # 各关节各自
        for j in range(7):
            print("    j%d  峰值 %5.2f°  RMS %5.2f°  末值 %5.2f°"
                  % (j + 1, np.degrees(np.abs(E[:, j]).max()),
                     np.degrees(np.sqrt((E[:, j] ** 2).mean())),
                     np.degrees(abs(E[-1, j]))))
        print()
        # 振铃检测：运动停下来后，实际速度是否还在振荡
        V = np.gradient(Q, dt, axis=0)
        tail = V[int(0.7 * cnt):]
        print("  后 30%% 段内各关节速度 RMS: %s"
              % " ".join("%.4f" % v for v in np.sqrt((tail ** 2).mean(axis=0))))

    # 直接测阶跃响应，估阻尼比
    print()
    print("=== 阶跃响应（ctrl 直接跳 2°，看超调）===")
    for j, name in enumerate(pp.ARM):
        pp.reset(m, d, qad)
        q_start = np.array(d.qpos[qad])
        d.ctrl[:7] = q_start
        for _ in range(int(0.5 / dt)):
            mujoco.mj_step(m, d)
        q_base = np.array(d.qpos[qad])
        d.ctrl[:7] = q_base
        d.ctrl[j] = q_base[j] + math.radians(2.0)
        rec = []
        for _ in range(int(1.5 / dt)):
            mujoco.mj_step(m, d)
            rec.append(d.qpos[qad[j]])
        rec = np.array(rec)
        delta = math.radians(2.0)
        overshoot = (rec.max() - q_base[j]) / delta - 1.0
        err_final = (rec[-1] - q_base[j]) / delta
        print("  %s  超调 %+6.1f%%   稳态到达 %5.1f%%" %
              (name, 100 * overshoot, 100 * err_final))


if __name__ == "__main__":
    raise SystemExit(main())
