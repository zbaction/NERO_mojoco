#!/usr/bin/env python3
"""解出 NERO 的抓取关键位姿 -> poses.json

问题背景
  joint7 的限位只有 ±97°，所以"完整 6DOF 姿态"经常无解。
  但抓取 10 cm 立方体并不需要任意姿态：需要的只是
    ① 两指指尖中点落在目标点（3 个约束）
    ② 工具 z 轴竖直向下（2 个约束）
    ③ 两指连线沿世界 ±y（1 个约束）—— 这样两个指尖正好夹住方块的 ±y 两面
  共 6 个约束、7 个关节，正好留一个冗余度。

做法
  多随机起点 + 带软限位处理的阻尼最小二乘。残差 = 位置(3) + 工具轴倾角(3)
  + 夹持面偏航角(1)，后两组各带权重。解完后在多个解里挑关节余量最充裕、
  且离上部待命姿态最近的那个，保证相邻位姿之间路径短、不绕身后。
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE / "nero_table_scene.xml"))
d = mujoco.MjData(m)
ARM = ["joint%d" % i for i in range(1, 8)]
EE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
PADL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_pad")
PADR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_pad")
BODL = m.geom_bodyid[PADL]
BODR = m.geom_bodyid[PADR]
QAD = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
DAD = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
LO = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][0] for j in ARM])
HI = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][1] for j in ARM])
JP = np.zeros((3, m.nv))
JR = np.zeros((3, m.nv))
JPL = np.zeros((3, m.nv))
JRL = np.zeros((3, m.nv))
JPR = np.zeros((3, m.nv))
JRR = np.zeros((3, m.nv))
DOWN = np.array([0.0, 0.0, -1.0])

W_TILT = 0.10     # 工具轴倾角权重（rad 与 m 的量纲折算）
W_YAW = 0.06      # 夹持面偏航权重（1 m 的位置误差 ≈ 16 rad 的偏航误差）


def fk(q):
    """正运动学：返回 (两指指尖中点, 末端旋转矩阵, padL 位置, padR 位置)。"""
    d.qpos[:] = 0
    for i, a in enumerate(QAD):
        d.qpos[a] = q[i]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mid = 0.5 * (d.geom_xpos[PADL] + d.geom_xpos[PADR])
    return (mid, d.xmat[EE].reshape(3, 3).copy(),
            d.geom_xpos[PADL].copy(), d.geom_xpos[PADR].copy())


def tilt_rad(R):
    z = R[:, 2]
    return math.atan2(np.linalg.norm(np.cross(z, DOWN)), float(np.dot(z, DOWN)))


def yaw_rad(padl, padr):
    """两指连线在 XY 平面内相对世界 +y 轴的偏角，值域 (-pi/2, pi/2]。"""
    a = float(padl[0] - padr[0])
    b = float(padl[1] - padr[1])
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return 0.0, a, b
    th = math.atan2(a, b)
    if th > math.pi / 2:
        th -= math.pi
    elif th <= -math.pi / 2:
        th += math.pi
    return th, a, b


def refine(q0, pt, iters=400, lam0=0.03, yaw_tgt=0.0):
    q = np.array(q0, float)
    lam = lam0
    best = (1e9, q.copy())
    for _ in range(iters):
        pm, R, pl, pr = fk(q)
        dp = pt - pm
        z = R[:, 2]
        ax = np.cross(z, DOWN)
        s = np.linalg.norm(ax)
        ang = math.atan2(s, float(np.dot(z, DOWN)))
        dr = ax / s * ang if s > 1e-9 else np.zeros(3)
        dy, a, b = yaw_rad(pl, pr)
        # 目标偏航：夹持面法向对准方块自己的朝向（方块可能绕 z 转过一个角度）
        dy = dy - yaw_tgt
        while dy > math.pi / 2:
            dy -= math.pi
        while dy <= -math.pi / 2:
            dy += math.pi

        # 代价：位置为主，倾角/偏航都必须压到 0（倾角单位 rad ≈ m 量级）
        c = (np.linalg.norm(dp) + 0.35 * ang + 0.010 * abs(dy)
             + (0.0 if (q - LO).min() > 0.05 and (HI - q).min() > 0.05 else 0.02))
        if c < best[0]:
            best = (c, q.copy())

        mujoco.mj_jacBody(m, d, JP, JR, EE)
        mujoco.mj_jac(m, d, JPL, JRL, pl, BODL)   # pad 点处的雅可比
        mujoco.mj_jac(m, d, JPR, JRR, pr, BODR)
        Jv = (JPL - JPR)[:, DAD]                  # d(padL-padR)/dq
        denom = a * a + b * b
        if denom > 1e-12:
            Jyaw = (b * Jv[0] - a * Jv[1]) / denom
        else:
            Jyaw = np.zeros(len(DAD))

        J = np.vstack([JP[:, DAD],
                       W_TILT * JR[:, DAD],
                       W_YAW * Jyaw[None, :]])
        err = np.concatenate([dp, W_TILT * dr, [W_YAW * (-dy)]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(7), err)
        qn = q + 0.8 * dq
        if (np.maximum(qn - HI, 0) + np.maximum(LO - qn, 0)).max() > 1e-9:
            ok = False
            for sc in (0.7, 0.5, 0.35, 0.2, 0.1, 0.05):
                t = q + 0.8 * sc * dq
                if (np.maximum(t - HI, 0) + np.maximum(LO - t, 0)).max() < 1e-9:
                    qn, ok = t, True
                    break
            if not ok:
                lam = min(0.5, lam * 2.0)
                continue
        if np.linalg.norm(dq) < 1e-11:
            break
        q = qn
        lam = max(0.004, lam * 0.97)
    return best[1]


def solve(pt, n_seed=6, shots=600, keep=4, seed=0, q_ref=None, yaw_tgt=0.0):
    """多起点求解，返回候选列表 [(score, q, perr, tilt, yaw_err)]。"""
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(n_seed):
        qs = rng.uniform(LO, HI, size=(shots, 7))
        sc = []
        for q in qs:
            pm, R, pl, pr = fk(q)
            th, _, _ = yaw_rad(pl, pr)
            e = (th - yaw_tgt + math.pi / 2) % math.pi - math.pi / 2
            s = (np.linalg.norm(pt - pm) + 0.12 * tilt_rad(R)
                 + 0.03 * abs(e))
            if q_ref is not None:
                s += 0.05 * np.linalg.norm(q - q_ref)
            sc.append(s)
        for i in np.argsort(sc)[:keep]:
            q = refine(qs[i], pt, yaw_tgt=yaw_tgt)
            pm, R, pl, pr = fk(q)
            perr = float(np.linalg.norm(pt - pm))
            tilt = tilt_rad(R)
            th, _, _ = yaw_rad(pl, pr)
            e = (th - yaw_tgt + math.pi / 2) % math.pi - math.pi / 2
            if perr < 2e-3 and tilt < math.radians(0.6):
                margin = float(np.minimum(q - LO, HI - q).min())
                score = -abs(e) * 3.0
                score -= 0.6 * (0.0 if margin > 0.12 else (0.12 - margin) / 0.12)
                if q_ref is not None:
                    # 强锚定参考姿态：IK 分支很多（joint5 正负两解、joint4 折叠解），
                    # 只给弱惩罚的话会选到"手臂绕到身后"的构型，关节空间要跑 146°，
                    # 动力学跟不过去。权重放大到 2.0，明确优先选靠近参考的那一支。
                    score -= 2.0 * float(np.linalg.norm(q - q_ref)) / math.sqrt(7)
                cands.append((score, q.copy(), perr, tilt, abs(e)))
    cands.sort(key=lambda c: -c[0])
    return cands


def pick(pt, seed=0, q_ref=None, yaw_tgt=0.0):
    cands = solve(pt, seed=seed, q_ref=q_ref, yaw_tgt=yaw_tgt)
    return cands[0] if cands else None


def cube_pose(name):
    """从场景里读方块的 (中心, 绕 z 的偏航角)。

    注意：freejoint 的位置就在 qpos 里，所以不能用 qpos[:]=0（会把它挪到原点），
    必须 mj_resetData 恢复 model 里的初始位姿。
    """
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    R = d.xmat[bid].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return d.xpos[bid].copy(), yaw


def basket_frame():
    """从场景里读篮子的 (中心 xy, 放置点 z)。

    放置点 z = 底板顶面 + 半个方块 + 2 mm 余量（避免方块与底板初始穿模）。
    """
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "basket")
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basket_floor")
    if bb < 0 or fg < 0:
        return None
    cx, cy = float(d.xpos[bb][0]), float(d.xpos[bb][1])
    floor_top = float(d.geom_xpos[fg][2]) + float(m.geom_size[fg][2])
    return (cx, cy), floor_top


CUBES = ("cube_red", "cube_green", "cube_blue")
CUBE_COLOR = {"cube_red": "red", "cube_green": "green", "cube_blue": "blue"}


def solve_task(cube, q_ref=None, verbose=True):
    """解某一个方块的一整套抓取-放置关键帧。

    返回 {关键帧名: 关节角(度)}。关键帧名带颜色后缀，例如
        red_hover / red_grasp / red_lift / red_transit / red_place_hover / red_place
    这样同一个场景可以同时存三套链条，运行时按颜色选一组。

    q_ref 必须是 ready 姿态：IK 有多个分支（joint5 可以正也可以负），
    不锚定的话红色/蓝色会解到"手臂绕到身后"的构型，关节空间轨迹要
    扫过 146°，动力学根本跟不过去（实测指尖偏 65~72 mm）。
    """
    c, yaw_c = cube_pose(cube)
    # 符号修正（重要）：cube_pose() 的 yaw_c 是绕 +z 的逆时针角，
    # 即 atan2(R[1,0], R[0,0])；而 yaw_rad() 用 atan2(padl.x-padr.x,
    # padl.y-padr.y) 量的是"从 +y 转向 +x"的顺时针角，两者符号相反。
    # 早期直接传 yaw_c，夹爪就朝反方向转，两指连线与方块面法向差 2×yaw：
    # 红 24°、绿 16°、蓝 40°。方块 80 mm 在这三个角度下的投影宽分别是
    # 105.6 / 99.0 / 112.7 mm，超过夹爪 100 mm 开度，下降时指尖顶住方块
    # 上棱角，joint2 力矩饱和到 -24 Nm，指尖卡在 z=0.834 下不去。
    # 所以目标偏航必须是 -yaw_c。
    yaw_tgt = -yaw_c
    (bx, by), floor_top = basket_frame()
    place = np.array([bx, by, floor_top + 0.040 + 0.002])
    place_hover = place + np.array([0.0, 0.0, 0.120])
    # 中转点取"方块 → 篮子"路径的中点，抬到篮子口以上，避免扫到侧壁
    transit = np.array([(c[0] + bx) / 2.0, (c[1] + by) / 2.0, floor_top + 0.230])
    JOB = [
        # 名字, 指尖中点目标, 夹持面目标偏航（注意是 -yaw_c，见上面符号说明）
        ("hover",       np.array([c[0], c[1], c[2] + 0.120]), yaw_tgt),
        ("grasp",       np.array([c[0], c[1], c[2]]), yaw_tgt),
        ("lift",        np.array([c[0], c[1], c[2] + 0.150]), yaw_tgt),
        ("transit",     transit, yaw_tgt),
        ("place_hover", place_hover, yaw_tgt),
        ("place",       place, yaw_tgt),
    ]
    out = {}
    q_prev = q_ref
    for k, (name, pt, yaw_tgt) in enumerate(JOB):
        r = pick(np.array(pt), seed=100 + k, q_ref=q_prev, yaw_tgt=yaw_tgt)
        if r is None:
            if verbose:
                print("  %-11s 无解" % name)
            continue
        _, q, perr, tilt, yw = r
        q_prev = q.copy()
        if verbose:
            print("  %-11s pos_err=%5.2f mm  tilt=%4.2f deg  夹持面偏航误差=%5.2f deg  "
                  "margin=%.1f deg"
                  % (name, perr * 1000, math.degrees(tilt), math.degrees(yw),
                     math.degrees(np.minimum(q - LO, HI - q).min())))
        out[name] = [round(math.degrees(v), 2) for v in q]
    return out


if __name__ == "__main__":
    # 直接从场景读三个方块的真实位姿（位置 + 绕 z 偏航），
    # 这样夹爪的夹持面永远对齐方块自己的两个侧面。
    cubes = {n: cube_pose(n) for n in CUBES}
    (bx, by), floor_top = basket_frame()
    print("篮子中心 (%.3f, %.3f)  底板顶面 z=%.3f" % (bx, by, floor_top))
    for n in CUBES:
        c, yaw = cubes[n]
        print("  %-11s 中心 [%.3f %.3f %.3f]  偏航 %+.1f°"
              % (n, *c, math.degrees(yaw)))

    out = {}
    # 1) 待命姿态：取绿方块上方，三种颜色共用
    cg, _ = cubes["cube_green"]
    r = pick(np.array([cg[0], cg[1], cg[2] + 0.295]), seed=100, q_ref=None, yaw_tgt=0.0)
    if r is None:
        raise SystemExit("待命姿态无解")
    _, q, perr, tilt, yw = r
    print("\nready       pos_err=%5.2f mm  tilt=%4.2f deg  margin=%.1f deg"
          % (perr * 1000, math.degrees(tilt),
             math.degrees(np.minimum(q - LO, HI - q).min())))
    out["ready"] = [round(math.degrees(v), 2) for v in q]

    # 2) 每个颜色一条独立的关键帧链，名字带颜色前缀；全部以 ready 为锚点
    q_ready = np.radians(np.array(out["ready"]))
    for n in CUBES:
        col = CUBE_COLOR[n]
        print("\n%s (%s)" % (col, n))
        chain = solve_task(n, q_ref=q_ready)
        for k, v in chain.items():
            out["%s_%s" % (col, k)] = v

    (HERE / "poses.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n写出 poses.json（%d 个位姿：ready + 3 色 × 6 个关键帧）" % len(out))
