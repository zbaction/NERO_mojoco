#!/usr/bin/env python3
"""诊断机械臂运动的流畅性（连续轨迹版）。

记录整条 pick-place 轨迹 **每一步** 的 qpos / ctrl / 指尖位置，然后算：
  1. 关节速度、加速度、加加速度（jerk）峰值
  2. "停顿"次数（速度掉到峰值 5% 以下且持续 > 0.1 s）与累计静止时长
  3. 末端（指尖中点）速度曲线 —— 看有没有锯齿/忽快忽慢
  4. 仿真时间 vs 墙钟时间
  5. ctrl 速度突变（>5 rad/s 的相邻步差值）出现的位置
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

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

    rec_q, rec_c, rec_p = [], [], []
    t_wall0 = time.time()

    # 拦截 mj_step，逐步记录（不改动 pick_place 的执行逻辑）
    orig_step = mujoco.mj_step

    def hooked(mm, dd):
        orig_step(mm, dd)
        rec_q.append(np.array(dd.qpos[qad]))
        rec_c.append(np.array(dd.ctrl[:7]))
        rec_p.append(0.5 * (dd.geom_xpos[padl] + dd.geom_xpos[padr]).copy())

    mujoco.mj_step = hooked
    try:
        for item in pp.plan_for(col):
            kind, keys, grip = item[0], item[1], item[2]
            hold = item[4] if len(item) > 4 else ()
            slow = item[5] if len(item) > 5 else ()
            if kind == "path":
                pp.move_path(m, d, qad, keys, grip, None, hold=hold, slow=slow)
            else:
                pp.dwell(m, d, qad, grip)
    finally:
        mujoco.mj_step = orig_step
    t_wall = time.time() - t_wall0

    Q = np.array(rec_q)
    C = np.array(rec_c)
    P = np.array(rec_p)
    n = len(Q)
    t_sim = n * dt
    print("方块 %s   仿真 %.2f s（%d 步）   墙钟 %.2f s   倍速 %.2fx"
          % (col, t_sim, n, t_wall, t_sim / t_wall))
    print()

    def stats(name, A, scale=1.0, unit=""):
        # 注意：只在**求导时**用原始单位，最后统一乘 scale，
        # 否则会像早期版本那样把 180/pi 乘两遍（报出的加速度大 57 倍）
        v = np.gradient(A, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        j = np.gradient(a, dt, axis=0)
        print("%-14s 速度峰值 %8.2f %s   加速度峰值 %9.1f %s/s    jerk 峰值 %10.0f %s/s²"
              % (name, np.abs(v).max() * scale, unit,
                 np.abs(a).max() * scale, unit,
                 np.abs(j).max() * scale, unit))
        return v * scale

    vq = stats("关节 qpos", Q, 180.0 / math.pi, "°/s")
    stats("关节 ctrl", C, 180.0 / math.pi, "°/s")
    vp = stats("指尖 pos", P, 1.0, "m/s")
    print()

    # 停顿检测：指尖速度 < 峰值 5% 且持续 > 0.1 s
    sp = np.linalg.norm(vp, axis=1)
    peak = sp.max()
    thr = 0.05 * peak
    quiet = sp < thr
    runs = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            runs.append((j - i, i))
            i = j
        else:
            i += 1
    long_runs = [(ln, st) for ln, st in runs if ln * dt > 0.10]
    print("指尖峰值 %.3f m/s   平均 %.3f m/s" % (peak, sp.mean()))
    print("停顿（<%.0f%% 峰值 = %.3f m/s，且 >0.1 s）：%d 次，累计 %.2f s"
          % (5, thr, len(long_runs), sum(ln for ln, _ in long_runs) * dt))
    for ln, st in long_runs:
        print("    t=%6.2f–%6.2f s   持续 %.2f s" % (st * dt, (st + ln) * dt, ln * dt))
    print()

    # ctrl 相邻步的最大跳变
    dc = np.abs(np.diff(C, axis=0)).max(axis=1) / dt
    idx = np.argsort(dc)[-6:][::-1]
    print("ctrl 速度最大的 6 个时刻：")
    for i in idx:
        print("  t=%6.3f s   速度 %.1f rad/s   关节 j%d"
              % ((i + 1) * dt, dc[i], int(np.argmax(np.abs(C[i + 1] - C[i]))) + 1))
    print()

    # 每段轨迹的时长与峰值速度
    print("逐段统计（关节空间，相对限速 pi / 1.25pi）：")
    tt = 0.0
    for item in pp.plan_for(col):
        kind, keys = item[0], item[1]
        if kind != "path":
            tt += pp.DWELL_SEC
            print("  dwell  夹爪动作                        t=%6.2f s" % tt)
            continue
        Qs = np.array([np.array([m.key(k).qpos[a] for a in qad]) if k is not None
                       else np.array([m.key(keys[1]).qpos[a] for a in qad]) for k in keys])
        lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][0]
                       for j in pp.ARM])
        hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][1]
                       for j in pp.ARM])
        hold = item[4] if len(item) > 4 else ()
        slow = item[5] if len(item) > 5 else ()
        hold_i = {i for i, k in enumerate(keys) if k is not None and k in set(hold)}
        slow_i = set()
        for k in slow:
            if k in keys and keys.index(k) > 0:
                slow_i.add(keys.index(k) - 1)
        samp, idx2, T, damp = pp._build_path(Qs, lo, hi, dt, hold=hold_i, slow=slow_i)
        dur = len(samp) * dt
        tt += dur
        print("  path   %-28s t=%6.2f s  时长 %5.2f s  damp=%.2f  %s"
              % (" → ".join(str(k) for k in keys), tt, dur, damp,
                 np.round(T, 2).tolist()))


if __name__ == "__main__":
    main()
