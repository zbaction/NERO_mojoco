#!/usr/bin/env python3
"""校验 nero_table_scene.xml：结构 / 限位 / 运动学 / 稳定性 / 抓取 / 渲染。

用法：
    python3 validate_nero.py             # 全量（含渲染）
    python3 validate_nero.py --no-render # 只算不渲染
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import mujoco
from PIL import Image

from depth_cam import DepthCamera
from depth_view import colormap as depth_colormap
from depth_view import measure as depth_measure
from depth_view import write_ply as depth_write_ply

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "nero_table_scene.xml"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

ARM = ["joint%d" % i for i in range(1, 8)]
LIMITS = {"joint1": (-157, 157), "joint2": (-102, 102), "joint3": (-160, 160),
          "joint4": (-60, 125), "joint5": (-160, 160), "joint6": (-44, 57),
          "joint7": (-97, 97)}
BODY_KG, GRIP_KG = 4.80, 0.50
TABLE_Z = 0.74                      # 桌面上表面高度
GRIP_OPEN, GRIP_CLOSE = 0.050, 0.0   # ctrl：0=完全闭合，0.055=全开（内间隙 2*ctrl）

ok = True


def check(cond, msg):
    global ok
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        ok = False
    return cond


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def main():
    render = "--no-render" not in sys.argv
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    EE = bid(m, "end_effector")
    PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
    PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
    print("模型 nbody=%d ngeom=%d njnt=%d nu=%d nq=%d"
          % (m.nbody, m.ngeom, m.njnt, m.nu, m.nq))

    def pad_mid():
        return 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])

    # ---------------- 1) 结构
    print("\n[1] 结构")
    check(all(jid(m, n) >= 0 for n in ARM), "7 个手臂关节齐全")
    check(all(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act") >= 0
              for n in ARM), "7 个手臂执行器绑定成功")
    for s in ("l", "r"):
        check(jid(m, "grip_" + s) >= 0, "夹爪关节 grip_%s 存在" % s)
        check(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_%s_act" % s) >= 0,
              "夹爪执行器 grip_%s_act 绑定成功" % s)

    body_mass = sum(m.body_mass[bid(m, n)] for n in
                    ("nero_base", "link1", "link2", "link3", "link4", "link5",
                     "link6", "link7", "end_effector"))
    grip_mass = sum(m.body_mass[bid(m, n)] for n in
                    ("gripper_base", "finger_l", "finger_r"))
    print("       本体(含法兰) %.3f kg   夹爪 %.3f kg   合计 %.3f kg"
          % (body_mass, grip_mass, body_mass + grip_mass))
    check(abs(body_mass - BODY_KG) < 0.02, "本体质量 %.3f kg 对齐手册 4.8 kg" % body_mass)
    check(abs(grip_mass - GRIP_KG) < 0.02, "夹爪质量 %.3f kg 对齐手册 0.5 kg" % grip_mass)

    # ---------------- 2) 限位 / 速度
    print("\n[2] 关节限位（对照 NERO 用户手册 1.2）")
    for j, (lo, hi) in LIMITS.items():
        r = m.jnt_range[jid(m, j)]
        g = (math.degrees(r[0]), math.degrees(r[1]))
        check(abs(g[0] - lo) < 1e-3 and abs(g[1] - hi) < 1e-3,
              "%-7s %7.1f~%6.1f°  手册 %6.1f~%6.1f°" % (j, g[0], g[1], lo, hi))

    # ---------------- 3) 运动学
    print("\n[3] 运动学与工作空间")
    qad = [m.jnt_qposadr[jid(m, n)] for n in ARM]
    mujoco.mj_forward(m, d)
    for name in ("ready", "red_grasp", "green_grasp", "blue_grasp"):
        if m.nkey and any(m.key(i).name == name for i in range(m.nkey)):
            k = m.key(name)
            d.qpos[:] = k.qpos
            mujoco.mj_forward(m, d)
            pm = pad_mid()
            zc = d.xmat[EE].reshape(3, 3)[:, 2]
            print("       %-11s 指尖中点 [%.3f %.3f %.3f]  工具z %s"
                  % (name, *pm, np.round(zc, 3)))

    d.qpos[:] = m.key("ready").qpos
    mujoco.mj_forward(m, d)
    p_ready = pad_mid().copy()
    check(p_ready[2] > 0.90, "ready 指尖高于桌面 0.15 m 以上 (z=%.3f)" % p_ready[2])

    rng = np.random.default_rng(0)
    pts = []
    for _ in range(400):
        d.qpos[:] = 0
        for i, a in enumerate(qad):
            lo, hi = m.jnt_range[jid(m, ARM[i])]
            d.qpos[a] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pts.append(pad_mid().copy())
    pts = np.array(pts)
    print("       随机采样指尖范围 x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]"
          % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
             pts[:, 2].min(), pts[:, 2].max()))
    check(pts[:, 2].max() - pts[:, 2].min() > 0.5, "竖直工作空间 > 0.5 m")
    check(np.ptp(pts[:, 0]) > 0.6, "水平伸展 > 0.6 m")

    # ---------------- 4) 稳定性
    print("\n[4] 稳定性（ready 保持 3 s，纯位置伺服 + 重力补偿）")
    q_ready = np.array([m.key("ready").qpos[a] for a in qad])
    d.qpos[:] = m.key("ready").qpos
    d.qvel[:] = 0
    d.ctrl[:7] = q_ready
    d.ctrl[7] = d.ctrl[8] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = np.degrees(np.array([d.qpos[a] for a in qad]) - q_ready)
    print("       各关节漂移(度): " + " ".join("%+.3f" % x for x in drift))
    check(np.abs(drift).max() < 0.5,
          "ready 保持漂移 < 0.5°（最大 %.4f°）" % np.abs(drift).max())
    for n in ("cube_red", "cube_green", "cube_blue"):
        b = bid(m, n)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n + "_geom")
        z0 = TABLE_Z + float(m.geom_size[g][2])
        check(abs(d.xpos[b][2] - z0) < 0.01,
              "%s 稳定在桌面 (z=%.3f，期望 %.3f)" % (n, d.xpos[b][2], z0))

    # ---------------- 5) 抓取
    print("\n[5] 抓取（ready -> hover -> grasp -> 闭合 -> 抬起）")
    green = bid(m, "cube_green")
    p_before = d.xpos[green].copy()

    def goto(name, grip, secs):
        q = np.array([m.key(name).qpos[a] for a in qad])
        d.ctrl[:7] = q
        d.ctrl[7] = d.ctrl[8] = grip
        for _ in range(int(secs / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return d.xpos[green].copy()

    goto("green_hover", GRIP_OPEN, 2.0)
    p_hover = pad_mid().copy()
    print("       hover 指尖中点 [%.3f %.3f %.3f]" % tuple(p_hover))
    goto("green_grasp", GRIP_OPEN, 2.5)
    p_grasp = pad_mid().copy()
    print("       grasp 指尖中点 [%.3f %.3f %.3f]（方块中心 [%.3f %.3f %.3f]）"
          % (*p_grasp, *p_before))
    check(np.linalg.norm(p_grasp - p_before) < 0.02,
          "指尖对准方块中心（偏差 %.1f mm）" % (1000 * np.linalg.norm(p_grasp - p_before)))

    n_before = d.ncon
    p_closed = goto("green_grasp", GRIP_CLOSE, 2.0)
    con = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
        if "finger" in g1 + g2:
            con.append((g1, g2, c.dist))
    print("       闭合后指尖接触对: %s" % (con if con else "无"))
    check(len(con) >= 2, "两指都碰到方块（%d 个接触）" % len(con))

    # 抬起：保持夹紧，把机械臂竖直提回 hover（+0.12 m）。
    # 注意抬 joint2 会让腕部下移，反而不动方块，所以用反向的 hover 位姿。
    q_lift = np.array([m.key("green_hover").qpos[a] for a in qad])
    d.ctrl[:7] = q_lift
    d.ctrl[7] = d.ctrl[8] = GRIP_CLOSE
    for _ in range(int(2.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p_lift = d.xpos[green].copy()
    dz = p_lift[2] - p_before[2]
    print("       抬起后 green z %+.4f m -> [%.3f %.3f %.3f]" % (dz, *p_lift))
    check(dz > 0.03, "夹住并抬起方块（Δz=%+.3f m）" % dz)

    # ---------------- 6) 渲染
    if render:
        print("\n[6] 渲染")
        d.qpos[:] = m.key("ready").qpos
        mujoco.mj_forward(m, d)
        rend = mujoco.Renderer(m, 720, 1280)
        for cname in ("overview", "arm_front", "arm_side", "top"):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
            opt = mujoco.MjvOption()
            rend.update_scene(d, camera=cam, scene_option=opt)
            img = rend.render().copy()
            Image.fromarray(img).save(OUT / ("nero_%s.png" % cname))

            opt2 = mujoco.MjvOption()
            opt2.geomgroup[2] = 0
            rend.update_scene(d, camera=cam, scene_option=opt2)
            img2 = rend.render().copy()
            mask = np.abs(img.astype(int) - img2.astype(int)).sum(axis=2) > 10
            frac = mask.mean()
            print("       %-10s 机械臂像素 %.2f%%（%d px）-> nero_%s.png"
                  % (cname, frac * 100, mask.sum(), cname))
            check(frac > 0.004, "%s 视角能看到机械臂（>0.4%%）" % cname)
        rend.close()

    # ---------------- 7) 腕部深度相机（eye-in-hand）
    print("\n[7] 腕部深度相机（RGB-D，eye-in-hand）")
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    check(cid >= 0, "场景里有 wrist_cam 相机")
    if cid >= 0:
        W, H = 848, 480
        check(W <= m.vis.global_.offwidth and H <= m.vis.global_.offheight,
              "深度图 %dx%d 不超过离屏帧缓冲 %dx%d"
              % (W, H, m.vis.global_.offwidth, m.vis.global_.offheight))

        # 7.1 安装位置：必须是 gripper_base 的子 body（跟着手走）
        cam_body_id = int(m.cam_bodyid[cid])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, cam_body_id)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        check(bname == "gripper_base", "相机挂在 gripper_base 上（实际 %r）" % bname)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key("ready").qpos
        mujoco.mj_forward(m, d)
        Rg = np.array(d.xmat[gid]).reshape(3, 3)
        p_grip = Rg.T @ (np.array(d.cam_xpos[cid]) - np.array(d.xpos[gid]))
        ax_grip = Rg.T @ (-np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2])
        print("       手眼外参（gripper_base 系）：光心 [%.3f %.3f %.3f] m，光轴 [%.3f %.3f %.3f]"
              % (*p_grip, *ax_grip))
        check(abs(np.linalg.norm(p_grip) - 0.097) < 0.02,
              "光心距夹爪原点 %.1f mm（95 mm 量级）" % (1000 * np.linalg.norm(p_grip)))
        check(ax_grip[2] > 0.55, "光轴在夹爪系里主要朝工具轴方向（z=%.2f）" % ax_grip[2])

        # 7.2 随姿态移动：三个姿态下相机世界坐标必须不同
        names = ("green_hover", "green_grasp", "green_place")
        pts = []
        for nm in names:
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key(nm).qpos
            mujoco.mj_forward(m, d)
            pts.append(np.array(d.cam_xpos[cid]))
        span = max(float(np.linalg.norm(a - b)) for i2, a in enumerate(pts) for b in pts[i2 + 1:])
        check(span > 0.15, "相机随夹爪移动（三姿态间最大位移 %.3f m）" % span)

        # 7.3 光轴朝下、能看到目标方块
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key("green_hover").qpos
        mujoco.mj_forward(m, d)
        R = np.array(d.cam_xmat[cid]).reshape(3, 3)
        axis = -R[:, 2]
        check(axis[2] < -0.30, "hover 时光轴朝下（z 分量 %.2f）" % axis[2])
        p0 = np.array(d.cam_xpos[cid])
        check(p0[2] > TABLE_Z + 0.05, "相机在桌面上方 %.3f m" % (p0[2] - TABLE_Z))
        gb = np.array(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_green")])
        v = gb - p0
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(axis, v / np.linalg.norm(v)))))))
        check(ang < 40.0, "绿方块在视野内（离轴 %.1f°）" % ang)

        # 7.4 三色各在自己的 hover 姿态做一次真实深度测量
        dc = DepthCamera(m, d, cam_name="wrist_cam", width=W, height=H, max_depth=1.0)
        fx, fy, cx_, cy_ = dc.intrinsics()
        print("       内参 fx=fy=%.2f cx=%.1f cy=%.1f" % (fx, cx_, cy_))
        lines = []
        for col in ("red", "green", "blue"):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key("%s_hover" % col).qpos
            mujoco.mj_forward(m, d)
            rgb, dep = dc.capture()
            if col == "green":
                Image.fromarray(rgb).save(OUT / "depth_rgb.png")
                Image.fromarray(depth_colormap(dep, lo=0.10, hi=0.90)).save(OUT / "depth_colormap.png")
                pts, pcols, _ = dc.pointcloud(dep, rgb)
                depth_write_ply(OUT / "depth_pointcloud.ply", pts, pcols)
                print("       点云 %d 点 -> out/depth_pointcloud.ply" % len(pts))
            body = "cube_" + col
            bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            gg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, body + "_geom")
            c_gt = np.array(d.xpos[bb])
            half = float(m.geom_size[gg][2])
            res = depth_measure(dc, rgb, dep, cols=(col,))[col]
            if not res.get("ok"):
                check(False, "%s 在 %s_hover 姿态下被深度相机测到（像素 %d）"
                      % (col, col, res.get("px", 0)))
                continue
            c = res["center"]
            dxy = float(np.hypot(c[0] - c_gt[0], c[1] - c_gt[1]))
            dz = float(c[2] - (c_gt[2] + half))
            lines.append((col, res["px"], c, c_gt, dxy, dz))
            check(dxy < 0.008 and abs(dz) < 0.012,
                  "%-5s %5d px 测得 [%.3f %.3f %.3f] Δxy=%.1f mm Δz=%.1f mm"
                  % (col, res["px"], *c, 1000 * dxy, 1000 * dz))
        dc.close()

        if len(lines) == 3:
            mean_err = sum(max(l[4], abs(l[5])) for l in lines) / 3.0
            print("       三色平均误差 %.1f mm（含顶面 z）" % (1000 * mean_err))
            check(mean_err < 0.006, "三色平均误差 < 6 mm")

    print("\n结果: %s" % ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
