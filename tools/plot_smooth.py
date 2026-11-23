#!/usr/bin/env python3
"""出"流畅性对比图"：指尖速度 / 关节速度 / 关节加速度曲线。

左列 = 旧版（11 段独立运动，段间停车）；右列 = 新版（连续 C2 轨迹）。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# WSL 里默认字体没有中文字形，会渲染成一堆方框 —— 显式挂上 Noto Sans CJK
for fp in ("/home/zhangb/.local/share/fonts/NotoSansCJKsc-Regular.otf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if pathlib.Path(fp).exists():
        matplotlib.font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = matplotlib.font_manager.FontProperties(
            fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = pathlib.Path.home() / "NERO"
sys.path.insert(0, str(HERE))
import pick_place as pp  # noqa: E402

OUT = HERE / "out"


def record(kind, col="green"):
    """kind: "old" 用 phases_for+move；"new" 用 plan_for+move_path/dwell。"""
    m, d, qad = pp.load()
    pp.reset(m, d, qad)
    dt = m.opt.timestep
    padl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
    padr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
    rq, rp, rc = [], [], []
    orig = mujoco.mj_step

    def hook(mm, dd):
        orig(mm, dd)
        rq.append(np.array(dd.qpos[qad]))
        rp.append(0.5 * (dd.geom_xpos[padl] + dd.geom_xpos[padr]).copy())
        rc.append(np.array(dd.ctrl[:7]))

    mujoco.mj_step = hook
    try:
        if kind == "old":
            for key, grip, secs, note in pp.phases_for(col):
                pp.move(m, d, qad, key, grip, secs)
        else:
            for item in pp.plan_for(col):
                k, keys, grip = item[0], item[1], item[2]
                hold = item[4] if len(item) > 4 else ()
                slow = item[5] if len(item) > 5 else ()
                if k == "path":
                    pp.move_path(m, d, qad, keys, grip, None, hold=hold, slow=slow)
                else:
                    pp.dwell(m, d, qad, grip)
    finally:
        mujoco.mj_step = orig
    Q = np.array(rq)
    P = np.array(rp)
    C = np.array(rc)
    vp = np.linalg.norm(np.gradient(P, dt, axis=0), axis=1)
    vq = np.degrees(np.abs(np.gradient(Q, dt, axis=0)))
    aq = np.degrees(np.abs(np.gradient(np.gradient(Q, dt, axis=0), dt, axis=0)))
    vc = np.degrees(np.abs(np.gradient(C, dt, axis=0)))
    t = np.arange(len(Q)) * dt
    return t, vp, vq, aq, vc


def main():
    OUT.mkdir(exist_ok=True)
    print("录制旧版（11 段独立运动）…")
    t0, vp0, vq0, aq0, vc0 = record("old")
    print("录制新版（连续 C2 轨迹）…")
    t1, vp1, vq1, aq1, vc1 = record("new")

    fig, ax = plt.subplots(4, 2, figsize=(13, 9), sharex="col")
    for c, (t, vp, vq, aq, vc, title) in enumerate([
            (t0, vp0, vq0, aq0, vc0, "旧版：11 段独立运动（段间停车）"),
            (t1, vp1, vq1, aq1, vc1, "新版：连续 C2 轨迹（中途不停车）")]):
        a = ax[:, c]
        a[0].plot(t, vp, lw=1.2, color="#1f77b4")
        a[0].set_ylabel("指尖速度 m/s")
        a[0].set_title("%s\n总时长 %.1f s" % (title, t[-1]), fontsize=10)
        a[1].plot(t, vq.max(axis=1), lw=1.0, color="#d62728")
        a[1].set_ylabel("关节速度 °/s")
        a[2].plot(t, aq.max(axis=1), lw=0.8, color="#2ca02c")
        a[2].set_ylabel("关节加速度 °/s²")
        a[3].plot(t, vc.max(axis=1), lw=1.0, color="#9467bd")
        a[3].set_ylabel("ctrl 速度 °/s")
        a[3].set_xlabel("时间 s")
        for x in a:
            x.grid(alpha=0.3, lw=0.5)
            x.margins(x=0.01)
    fig.suptitle("NERO 抓取-放置：运动流畅性对比（绿方块）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "smooth_compare.png"
    fig.savefig(p, dpi=130)
    print("写出 %s" % p)

    print()
    print("%-22s %10s %10s" % ("指标", "旧版", "新版"))
    rows = [
        ("总时长 s", t0[-1], t1[-1]),
        ("指尖速度峰值 m/s", vp0.max(), vp1.max()),
        ("关节速度峰值 °/s", vq0.max(), vq1.max()),
        ("关节加速度峰值 °/s²", aq0.max(), aq1.max()),
        ("ctrl 速度峰值 °/s", vc0.max(), vc1.max()),
    ]
    for name, a, b in rows:
        print("%-22s %10.2f %10.2f" % (name, a, b))
    # 静止占比
    for name, vp, t in (("旧版", vp0, t0), ("新版", vp1, t1)):
        thr = 0.05 * vp.max()
        print("%s 静止占比 %.1f%%（阈值 %.3f m/s）"
              % (name, 100.0 * (vp < thr).mean(), thr))


if __name__ == "__main__":
    main()
