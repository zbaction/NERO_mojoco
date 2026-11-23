#!/usr/bin/env python3
"""NERO 抓取-放置任务：把指定颜色的方块从桌面搬进收纳篮。

任务序列（每个阶段都伺服到 poses.json 里对应的关键帧）
    ready → <col>_hover → <col>_grasp → [闭合] → <col>_lift
          → <col>_transit → <col>_place_hover → <col>_place
          → [张开] → <col>_place_hover → <col>_transit → ready

用法：
    python3 pick_place.py                  # 抓绿方块（默认）
    python3 pick_place.py --cube red       # 抓红方块
    python3 pick_place.py --cube blue      # 抓蓝方块
    python3 pick_place.py --cube green --viewer   # 带交互窗口
    python3 pick_place.py --cube red --check      # 只判定成功与否（退出码 0/1）
    python3 pick_place.py --all            # 依次把三块都搬进篮子
"""
from __future__ import annotations

import argparse
import math
import pathlib
import time

import numpy as np

import mujoco

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

ARM = ["joint%d" % i for i in range(1, 8)]
GRIP_OPEN, GRIP_CLOSE = 0.050, 0.0

COLORS = ("red", "green", "blue")
CUBE_OF = {"red": "cube_red", "green": "cube_green", "blue": "cube_blue"}
LABEL = {"red": "红方块", "green": "绿方块", "blue": "蓝方块"}


def phases_for(col):
    """某颜色的 11 段动作表：(关键帧, 夹爪 ctrl, 时长 s, 说明)。"""
    name = LABEL[col]
    return [
        ("%s_hover" % col,       GRIP_OPEN,  2.0, "移到%s正上方" % name),
        ("%s_grasp" % col,       GRIP_OPEN,  2.5, "下降到%s中心" % name),
        ("%s_grasp" % col,       GRIP_CLOSE, 1.5, "闭合夹爪"),
        ("%s_lift" % col,        GRIP_CLOSE, 2.0, "竖直提起 150 mm"),
        ("%s_transit" % col,     GRIP_CLOSE, 2.5, "平移到篮子上方"),
        ("%s_place_hover" % col, GRIP_CLOSE, 2.5, "到篮子正上方 120 mm"),
        ("%s_place" % col,       GRIP_CLOSE, 2.5, "下降把%s放进篮子" % name),
        ("%s_place" % col,       GRIP_OPEN,  1.5, "张开夹爪释放"),
        ("%s_place_hover" % col, GRIP_OPEN,  2.0, "退出篮子"),
        ("%s_transit" % col,     GRIP_OPEN,  2.0, "往回走"),
        ("ready",                GRIP_OPEN,  2.5, "回到待命姿态"),
    ]


PHASES = phases_for("green")   # 兼容旧代码 / 诊断脚本


def load():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
    return m, d, qad


def reset(m, d, qad):
    d.qpos[:] = m.key("ready").qpos
    d.qvel[:] = 0
    d.ctrl[:7] = [m.key("ready").qpos[a] for a in qad]
    d.ctrl[7] = d.ctrl[8] = GRIP_OPEN
    mujoco.mj_forward(m, d)


def goto(m, d, qad, key, grip, secs, viewer=None):
    """阶跃伺服到关键帧（保留给自检用）。"""
    q = np.array([m.key(key).qpos[a] for a in qad])
    d.ctrl[:7] = q
    d.ctrl[7] = d.ctrl[8] = grip
    n = int(secs / m.opt.timestep)
    for i in range(n):
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()


def move(m, d, qad, key, grip, secs, viewer=None):
    """余弦平滑地伺服到关键帧。

    阶跃式 ctrl 会让关节在第一步就要求满速，末端加速度可达 15 m/s² 量级，
    指尖与方块之间会蹭掉 4~10 mm（实测每段搬运累计 23 mm，方块最终偏离
    篮心 25 mm）。用 s(t)=0.5(1-cos(pi t/T)) 过渡，起停加速度为 0，
    滑移降到 1 mm 以内。
    """
    q0 = np.array([d.qpos[a] for a in qad], dtype=float)
    q1 = np.array([m.key(key).qpos[a] for a in qad], dtype=float)
    g0 = float(d.ctrl[7])
    g1 = float(grip)
    n = max(1, int(secs / m.opt.timestep))
    for i in range(n):
        s = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n))
        d.ctrl[:7] = q0 + (q1 - q0) * s
        d.ctrl[7] = d.ctrl[8] = g0 + (g1 - g0) * s
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()
    # 收尾：精确停在目标上并保持一小段，把残余跟踪误差收敛掉
    d.ctrl[:7] = q1
    d.ctrl[7] = d.ctrl[8] = float(grip)
    for i in range(int(0.4 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()


def pad_mid(m, d):
    pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
    pr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
    return 0.5 * (d.geom_xpos[pl] + d.geom_xpos[pr])


def basket_box(m, d):
    """篮子的内腔 (中心 xy, 底板顶面 z, 内半宽)。"""
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "basket")
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basket_floor")
    cx, cy = float(d.xpos[bb][0]), float(d.xpos[bb][1])
    floor_top = float(d.geom_xpos[fg][2]) + float(m.geom_size[fg][2])
    side = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basket_wall_xp")
    # 内表面 = 壁中心相对篮子中心的 x 偏移 - 壁半厚
    inner = (float(d.geom_xpos[side][0]) - cx) - float(m.geom_size[side][0])
    return (cx, cy), floor_top, inner


def cube_in_basket(m, d, cube):
    """判定方块是否稳稳落在篮子内腔里。"""
    (bx, by), floor_top, inner = basket_box(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, cube + "_geom")
    p = d.xpos[b]
    half = float(m.geom_size[g][2])
    ok_xy = abs(p[0] - bx) < inner - half and abs(p[1] - by) < inner - half
    ok_z = abs(p[2] - (floor_top + half)) < 0.012
    upright = d.xmat[b].reshape(3, 3)[2, 2] > 0.97
    return ok_xy and ok_z and upright, p, (bx, by), floor_top, inner


def run(m, d, qad, col="green", viewer=None, verbose=True):
    reset(m, d, qad)
    cube = CUBE_OF[col]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    p0 = d.xpos[bid].copy()
    (bx, by), floor_top, inner = basket_box(m, d)
    if verbose:
        print("任务：把%s [%.3f %.3f %.3f] 搬进篮子 [%.3f %.3f]（内腔 %.0f mm）"
              % (LABEL[col], *p0, bx, by, 2000 * inner))
        print()
        print("%-6s %-26s %-26s %s" % ("阶段", "方块位置", "指尖中点", "说明"))
    for i, (key, grip, secs, note) in enumerate(phases_for(col)):
        move(m, d, qad, key, grip, secs, viewer)
        p = d.xpos[bid].copy()
        pm = pad_mid(m, d)
        if verbose:
            print("%-6d [%.3f %.3f %.3f]        [%.3f %.3f %.3f]      %s"
                  % (i + 1, *p, *pm, note))
    ok, p, bc, ft, inner = cube_in_basket(m, d, cube)
    if verbose:
        print()
        print("最终：%s [%.3f %.3f %.3f]" % (LABEL[col], *p))
        print("      篮子中心 (%.3f, %.3f)  底板顶面 z=%.3f  内半宽 %.3f"
              % (bc[0], bc[1], ft, inner))
        print("      偏离中心 %.1f mm，底面高差 %.1f mm"
              % (1000 * math.hypot(p[0] - bc[0], p[1] - bc[1]),
                 1000 * abs(p[2] - (ft + 0.040))))
        print("结果：%s" % ("成功，%s在篮子内" % LABEL[col] if ok else "失败"))
    return ok


def shoot(m, d, name, look, dist, az, el, fovy=42, w=1280, h=720):
    r = mujoco.Renderer(m, h, w)
    m.vis.global_.fovy = fovy
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    r.update_scene(d, cam)
    from PIL import Image
    Image.fromarray(r.render()).save(OUT / ("%s.png" % name))
    r.close()
    return name


def shots(m, d, qad, col):
    reset(m, d, qad)
    cube = CUBE_OF[col]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    # 图名带颜色前缀，否则三种颜色会互相覆盖
    marks = {"%s_grasp" % col: ("pp_%s_1_grasped" % col, (0.08, 0.02, 0.92), 0.95, 55, -14),
             "%s_lift" % col: ("pp_%s_2_lifted" % col, (0.04, 0.06, 1.00), 1.05, 55, -14),
             "%s_transit" % col: ("pp_%s_3_transit" % col, (-0.00, 0.22, 1.00), 1.10, 55, -16),
             "%s_place_hover" % col: ("pp_%s_4_above_basket" % col, (-0.02, 0.32, 0.95), 0.95, 62, -18),
             # 篮子侧壁 90 mm 高、方块 80 mm 高 —— 低机位会被完全遮住，
             # 必须用俯视（el=-52）+ 近距离才能看到方块躺在篮底
             "%s_place" % col: ("pp_%s_5_placed" % col, (-0.04, 0.42, 0.80), 0.62, 62, -52)}
    shot = set()
    for i, (key, grip, secs, note) in enumerate(phases_for(col)):
        move(m, d, qad, key, grip, secs)
        if key in marks and key not in shot:
            nm, look, dist, az, el = marks[key]
            shoot(m, d, nm, look, dist, az, el)
            shot.add(key)
            print("  写出 out/%s.png   %s z=%.4f"
                  % (nm, LABEL[col], d.xpos[bid][2]))
    # 收尾全景 + 篮子俯视特写
    shoot(m, d, "pp_%s_6_done" % col, (-0.06, 0.20, 0.88), 1.15, 60, -30)
    shoot(m, d, "pp_%s_7_basket_top" % col, (-0.04, 0.42, 0.80), 0.50, 62, -68)
    print("  写出 out/pp_%s_6_done.png / pp_%s_7_basket_top.png" % (col, col))
    ok, p, bc, ft, inner = cube_in_basket(m, d, cube)
    print()
    print("最终：%s [%.3f %.3f %.3f]" % (LABEL[col], *p))
    print("      偏离篮心 %.1f mm  底面高差 %.1f mm"
          % (1000 * math.hypot(p[0] - bc[0], p[1] - bc[1]),
             1000 * abs(p[2] - (ft + 0.040))))
    print("结果：%s" % ("成功，%s在篮子内" % LABEL[col] if ok else "失败"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="NERO 抓取-放置：把指定颜色的方块搬进篮子")
    ap.add_argument("--cube", choices=COLORS, default="green",
                    help="要抓的方块颜色（默认 green）")
    ap.add_argument("--all", action="store_true", help="依次搬红、绿、蓝三块")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--check", action="store_true", help="只判定成功与否")
    ap.add_argument("--shots", action="store_true", help="出过程图到 out/")
    a = ap.parse_args()

    m, d, qad = load()

    if a.viewer:
        import mujoco.viewer as _viewer
        import view_scene as vs
        cols = list(COLORS) if a.all else [a.cube]
        with _viewer.launch_passive(m, d) as v:
            vs.frame_camera(v, m, d, az=60, el=-20)
            ok = True
            for col in cols:
                ok = run(m, d, qad, col, viewer=v) and ok
            print("窗口保持中，关闭窗口退出…")
            while v.is_running():
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(m.opt.timestep)
        return 0 if ok else 1

    if a.shots:
        cols = list(COLORS) if a.all else [a.cube]
        ok = True
        for col in cols:
            ok = shots(m, d, qad, col) and ok
        return 0 if ok else 1

    if a.check:
        cols = list(COLORS) if a.all else [a.cube]
        allok = True
        for col in cols:
            run(m, d, qad, col, verbose=False)
            ok, p, bc, ft, inner = cube_in_basket(m, d, CUBE_OF[col])
            dxy = math.hypot(p[0] - bc[0], p[1] - bc[1])
            dz = abs(p[2] - (ft + 0.040))
            allok = allok and ok
            print("%-6s %s 落点 [%.3f %.3f %.3f]  偏离篮心 %.1f mm（限 %.1f）"
                  "  高差 %.1f mm（限 12.0）  -> %s"
                  % (col, LABEL[col], *p, 1000 * dxy, 1000 * (inner - 0.040),
                     1000 * dz, "PASS" if ok else "FAIL"))
        print("总结果：%s" % ("PASS" if allok else "FAIL"))
        return 0 if allok else 1

    if a.all:
        ok = True
        for col in COLORS:
            ok = run(m, d, qad, col, verbose=True) and ok
            print()
        return 0 if ok else 1

    return 0 if run(m, d, qad, a.cube, verbose=True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
