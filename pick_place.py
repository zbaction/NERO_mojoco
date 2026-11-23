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
GRIP_JOINTS = ["grip_l", "grip_r"]
GRIP_OPEN, GRIP_CLOSE = 0.050, 0.0

COLORS = ("red", "green", "blue")
CUBE_OF = {"red": "cube_red", "green": "cube_green", "blue": "cube_blue"}
LABEL = {"red": "红方块", "green": "绿方块", "blue": "蓝方块"}


def phases_for(col):
    """某颜色的 11 段动作表：(关键帧, 夹爪 ctrl, 时长 s, 说明)。

    **仅用于诊断脚本**（`tools/diag_*.py`）。实际执行走 `plan_for()`：
    那里把相邻途经点串成一条连续轨迹，不再每段都停车。
    """
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


def plan_for(col):
    """某颜色的执行计划：3 条连续轨迹 + 2 次夹爪动作。

    与 `phases_for()` 的区别是**相邻途经点串成一条连续轨迹**：机械臂
    经过 lift / transit 这类中途点时不停车、不减速，只有真正需要
    改变夹爪状态的两个点（grasp、place）才停顿。

    但**接近段必须在 hover 处把速度降到 0**：如果带着速度冲过 hover
    直接扎进方块，指尖会从侧面撞上方棱，蓝块（+20° 偏航、投影宽
    112.7 mm）直接被撞倒。所以 hover / place_hover 设为"零速途经点"，
    grasp / place 所在段额外放慢 —— 这是"落点要稳"，与"中途别停车"
    并不矛盾。

    每项：("path", [关键帧,...], 夹爪, 说明) 或 ("dwell", [], 夹爪, 说明)
    列表首项用 None 表示"当前姿态"。
    """
    name = LABEL[col]
    hv = "%s_hover" % col
    g, l, t = "%s_grasp" % col, "%s_lift" % col, "%s_transit" % col
    ph, pl = "%s_place_hover" % col, "%s_place" % col
    return [
        ("path",  [None, hv, g], GRIP_OPEN,  "接近并下降到%s中心" % name,
         {hv}, {g}),
        ("dwell", [], GRIP_CLOSE, "闭合夹爪"),
        ("path",  [g, l, t, ph, pl], GRIP_CLOSE, "提起 → 平移 → 放进篮子",
         {ph}, {pl}),
        ("dwell", [], GRIP_OPEN, "张开夹爪释放"),
        ("path",  [pl, ph, t, "ready"], GRIP_OPEN, "退出篮子 → 回到待命",
         set(), {pl}),
    ]


# --- 连续轨迹 ---------------------------------------------------------------
# 关节限速与 build_nero.py 的 JOINT_VEL 一致：j1~j3 = pi，j4~j7 = 1.25*pi
JVEL = np.array([math.pi] * 3 + [1.25 * math.pi] * 4)
JV_PEAK = 1.05          # 轨迹允许的峰值关节速度（rad/s），约为限速的 17~33%
V_NOM = 0.72            # 规划目标速度（rad/s）；S 形曲线的峰值约为它的 1.5~1.9 倍
T_SEG_MIN = 0.45        # 单段最短时长（s）
DWELL_SEC = 0.55        # 夹爪开/合停留时长（s）
T_SCALE_MAX = 6.0       # 速度超限时最多把整条轨迹放慢到几倍
SLOW_SEG = 2.2          # 需要精确落点的段（下降夹取、下放）放慢的倍数


def _hermite(qa, qb, ma, mb, T, n):
    """三次 Hermite 采样：给定两端位置与切向速度，返回 n 个点（不含末端）。

    注意：三次 Hermite 只保证 C1（速度连续），加速度在途经点会突变，
    高速时看得出来"抽一下"。实际采样走 `_hermite5()`，这个函数只在
    `hold` 退化路径里用不到，留着便于对比/诊断。
    """
    s = (np.arange(n, dtype=float) / n)[:, None]
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00 * qa + h10 * T * ma + h01 * qb + h11 * T * mb


def _hermite5(qa, qb, ma, mb, aa, ab, T, n):
    """五次 Hermite：同时匹配两端的位置、速度、加速度 → 整条曲线 C2。

    三次 Hermite 只在节点上做到速度连续，加速度是分段的，节点处会
    跳变（jerk 无穷大），视觉上就是"到点抽一下"。五次版本让加速度
    也连续，起步/停车都是 S 形速度曲线，动作平滑得多。
    """
    s = (np.arange(n, dtype=float) / n)[:, None]
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    h0 = 1 - 10 * s3 + 15 * s4 - 6 * s5
    h1 = s - 6 * s3 + 8 * s4 - 3 * s5
    h2 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
    h3 = 10 * s3 - 15 * s4 + 6 * s5
    h4 = -4 * s3 + 7 * s4 - 3 * s5
    h5 = 0.5 * s3 - s4 + 0.5 * s5
    return (h0 * qa + h1 * T * ma + h2 * T * T * aa
            + h3 * qb + h4 * T * mb + h5 * T * T * ab)


def _sample_path(Q, T, damp, dt, hold=()):
    """给定每段时长 T 与中间切向衰减 damp，采样整条轨迹。

    hold 里的下标表示该途经点**必须零速、零加速度通过**（像刹车点），
    用于"先到 hover 停稳、再竖直下降夹取"这类需要精确落点的地方。

    途经点速度用 Catmull-Rom 的时域版本，加速度用相邻两段弦加速度的
    差分（节点二阶导），两者都参与插值 → 整条轨迹 C2 连续。

    返回 (样本矩阵, 各途经点下标)。
    """
    nq = len(Q)
    knots = np.concatenate([[0.0], np.cumsum(T)])
    M = np.zeros_like(Q)
    A = np.zeros_like(Q)
    if damp > 0.0:
        # 节点速度：相邻两点的弦速度
        for i in range(1, nq - 1):
            if i in hold:
                continue
            M[i] = damp * (Q[i + 1] - Q[i - 1]) / (knots[i + 1] - knots[i - 1])
        # 节点加速度：两端弦加速度之差 / 两端时长的平均
        for i in range(1, nq - 1):
            if i in hold:
                continue
            Ta = max(T[i - 1], 1e-6)
            Tb = max(T[i], 1e-6)
            va = (Q[i] - Q[i - 1]) / Ta
            vb = (Q[i + 1] - Q[i]) / Tb
            A[i] = damp * (vb - va) / (0.5 * (Ta + Tb))

    parts, idx = [], []
    for i in range(nq - 1):
        n = max(2, int(round(T[i] / dt)))
        idx.append(sum(len(p) for p in parts))
        parts.append(_hermite5(Q[i], Q[i + 1], M[i], M[i + 1],
                               A[i], A[i + 1], T[i], n))
    idx.append(sum(len(p) for p in parts))
    samples = np.vstack(parts + [Q[-1][None, :]])
    return samples, idx


def _build_path(Q, lo, hi, dt, hold=(), slow=()):
    """按途经点造一条连续轨迹，返回 (样本, 各途经点下标, 每段时长)。

    切向用 Catmull-Rom 的时域版本：端点速度为 0，中间点取相邻两点的
    弦速度，因此**经过中间途经点时不减速**（这是"流畅"的关键）。

    hold: 必须零速通过的途经点下标（刹车点）。
    slow: 需要放慢的**段**下标（0-based，段 i = Q[i]→Q[i+1]）。

    两级重试：
      ① 关节限位越界 → 逐步衰减中间切向（1.0 → 0.6 → 0.35 → 0.2 → 0.1 → 0）
      ② 关节速度超限 → 整条轨迹按比例放慢，再重试。
    """
    hold = set(hold)
    slow = set(slow)
    Ls = np.abs(np.diff(Q, axis=0)).max(axis=1)
    T = np.maximum(T_SEG_MIN, 1.6 * Ls / V_NOM)
    for i in slow:
        if 0 <= i < len(T):
            T[i] *= SLOW_SEG

    for _round in range(12):
        grew = False
        for damp in (1.0, 0.6, 0.35, 0.2, 0.1, 0.0):
            samples, idx = _sample_path(Q, T, damp, dt, hold)
            if (np.maximum(samples - hi, 0) + np.maximum(lo - samples, 0)).max() > 1e-9:
                continue                      # 越限，换更保守的切向
            vmax = float(np.abs(np.gradient(samples, dt, axis=0)).max())
            if vmax <= JV_PEAK:
                return samples, idx, T, damp
            # 速度超限：整条放慢，重新走一遍
            T = T * max(1.15, vmax / JV_PEAK)
            grew = True
            break
        if not grew:
            break
        if T.max() / T_SEG_MIN > T_SCALE_MAX:
            break
    raise RuntimeError("无法构造无越限轨迹（Q 的关节跳变过大？）")


def _step_loop(m, d, dt, nsteps, ctrl_of, viewer=None, t_ref=None):
    """执行 nsteps 步；viewer 模式下按墙钟节拍同步，保证真实速度播放。

    早期版本在 --viewer 里是"能跑多快跑多快"（实测 82 倍速），
    画面上机械臂一闪而过、看着很卡。这里改成实时节拍。
    """
    if viewer is None:
        for i in range(nsteps):
            d.ctrl[:] = ctrl_of(i)
            mujoco.mj_step(m, d)
        return
    dec = max(1, int(round((1.0 / 60.0) / dt)))     # 渲染约 60 Hz
    if t_ref is None:
        t_ref = [time.perf_counter()]
    for i in range(nsteps):
        d.ctrl[:] = ctrl_of(i)
        mujoco.mj_step(m, d)
        if (i + 1) % dec == 0 or i == nsteps - 1:
            target = t_ref[0] + (i + 1) * dt
            wait = target - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            viewer.sync()



PHASES = phases_for("green")   # 兼容旧代码 / 诊断脚本


def load():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qad = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM]
    return m, d, qad


def reset(m, d, qad):
    """把机械臂复位到 ready —— **只动机械臂，不动方块**。

    早期版本写的是 d.qpos[:] = m.key("ready").qpos，这会把所有 freejoint
    一起重置。后果：`--all` 依次搬三块时，搬第二块开头的 reset 会把
    第一块已经从篮子里"搬回"桌面（实测 green 从 [-0.04 0.42 0.788]
    变回 [0.18 0 0.78]）。所以这里只写 7 个臂关节 + 两根手指，
    三个方块的 freejoint 位置保持原样。
    """
    k = m.key("ready").qpos
    gq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
          for n in GRIP_JOINTS]
    d.qpos[qad] = k[qad]
    d.qpos[gq] = k[gq]
    d.qvel[:] = 0
    d.ctrl[:7] = k[qad]
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


def move_path(m, d, qad, keys, grip, viewer=None, hold=(), slow=(), settle=0.15):
    """沿一串关键帧走一条**连续**轨迹，末尾短暂按住让误差收敛。

    keys 里可以有 None，表示"从当前姿态出发"。
    hold: 必须零速通过的途经点**名字**集合（刹车点，用于精确落点）。
    slow: 需要放慢的**段**下标集合（0-based）。
    settle: 末尾按住时长（s）。后面紧跟 dwell 时用小值即可，
            否则夹爪动作期间本来就在原地等。

    返回 (采样矩阵, 各途经点下标, 每段时长)。
    """
    dt = m.opt.timestep
    Q = []
    for k in keys:
        if k is None:
            Q.append(np.array([d.qpos[a] for a in qad], dtype=float))
        else:
            Q.append(np.array([m.key(k).qpos[a] for a in qad], dtype=float))
    Q = np.array(Q)
    lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][0]
                   for j in ARM])
    hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][1]
                   for j in ARM])
    hold_i = {i for i, k in enumerate(keys) if k is not None and k in set(hold)}
    # slow 传的是"段终点关键帧名"，换算成段下标（段 i = Q[i] → Q[i+1]）
    slow_i = set()
    for k in slow:
        if k in keys:
            i = keys.index(k)
            if i > 0:
                slow_i.add(i - 1)
    samp, idx, T, _M = _build_path(Q, lo, hi, dt, hold=hold_i, slow=slow_i)
    n = len(samp)
    g0 = float(d.ctrl[7])
    g1 = float(grip)
    # 夹爪在前 30% 时间内走完，避免和臂运动末期耦合
    ng = max(1, int(0.30 * n))
    # 起点从**当前 ctrl** 平滑接管到轨迹首点（避免段边界留 0.4° 台阶）
    c0 = np.array(d.ctrl[:7], dtype=float)
    nb = max(2, int(0.15 / dt))

    def ctrl_of(i):
        c = np.empty(m.nu)
        if i < nb:
            s = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / nb))
            c[:7] = c0 + (samp[i] - c0) * s
        else:
            c[:7] = samp[i]
        if i < ng:
            s = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / ng))
            c[7] = c[8] = g0 + (g1 - g0) * s
        else:
            c[7] = c[8] = g1
        return c

    t0 = time.perf_counter()
    _step_loop(m, d, dt, n, ctrl_of, viewer, t_ref=[t0])
    # 末尾按住一小段，把残余跟踪误差压下去。
    # 后面紧跟 dwell（原地夹爪动作）时不用长等 —— 夹爪动作期间本来就在原地。
    nhold = max(1, int(settle / dt))

    def ctrl_hold(i):
        c = np.empty(m.nu)
        c[:7] = Q[-1]
        c[7] = c[8] = g1
        return c

    _step_loop(m, d, dt, nhold, ctrl_hold, viewer,
               t_ref=[t0 + n * dt])
    return samp, idx, T


def dwell(m, d, qad, grip, secs=DWELL_SEC, viewer=None):
    """在当前位置停住，只改夹爪（张开/闭合）。

    关节目标从**上一步的 ctrl** 平滑过渡到实测 qpos（0.2 s 余弦混合）。
    直接用 qpos 当 ctrl 会在段边界留一个小台阶（实测 0.44°），
    混合一下就没有任何跳变了。
    """
    dt = m.opt.timestep
    n = max(1, int(secs / dt))
    nb = max(2, int(0.20 / dt))
    g0 = float(d.ctrl[7])
    c0 = np.array(d.ctrl[:7], dtype=float)
    q_hold = np.array([d.qpos[a] for a in qad], dtype=float)

    def ctrl_of(i):
        c = np.empty(m.nu)
        if i < nb:
            s = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / nb))
            c[:7] = c0 + (q_hold - c0) * s
        else:
            c[:7] = q_hold
        s = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n))
        c[7] = c[8] = g0 + (grip - g0) * s
        return c

    _step_loop(m, d, dt, n, ctrl_of, viewer, t_ref=[time.perf_counter()])


def qad_of(m):
    return [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM]


def move(m, d, qad, key, grip, secs, viewer=None):
    """余弦平滑地伺服到关键帧（单段，留给诊断脚本用）。

    **主流程已不再用它**：单段之间要停车，11 段下来一共静止 8.11 s，
    看着就是"一顿一顿"。实际执行走 `run()` 里的 `plan_for()` +
    `move_path()`，相邻途经点串成连续轨迹。
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


def slot_xy(m, d, col):
    """某个颜色的放置格位（场景里的 site）。"""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "slot_%s" % col)
    if sid < 0:
        raise RuntimeError("场景里没有 slot_%s，请先跑 build_nero.py" % col)
    return float(d.site_xpos[sid][0]), float(d.site_xpos[sid][1])


def cube_in_basket(m, d, cube, col=None):
    """判定方块是否稳稳落在**它自己的格位**上。

    篮子放大到 280 mm 后，三个颜色各占一个格位（间距 120 mm）。
    如果还按"离篮心 < 内半宽"判，红/绿块落在自己格位里也会被判失败，
    所以这里改成跟自己的 slot 比。
    """
    (bx, by), floor_top, inner = basket_box(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, cube + "_geom")
    p = d.xpos[b]
    half = float(m.geom_size[g][2])
    # 格位中心（没传颜色就从 cube 名字里推）
    if col is None:
        col = cube.replace("cube_", "")
    tx, ty = slot_xy(m, d, col)
    ok_xy = abs(p[0] - tx) < 0.020 and abs(p[1] - ty) < 0.020
    ok_z = abs(p[2] - (floor_top + half)) < 0.012
    # 还要真的在内腔里（防止 site 写错把格位放到墙外）
    in_box = abs(p[0] - bx) < inner - half and abs(p[1] - by) < inner - half
    upright = d.xmat[b].reshape(3, 3)[2, 2] > 0.97
    return (ok_xy and ok_z and in_box and upright), p, (tx, ty), floor_top, inner


def run(m, d, qad, col="green", viewer=None, verbose=True):
    reset(m, d, qad)
    cube = CUBE_OF[col]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    p0 = d.xpos[bid].copy()
    (bx, by), floor_top, inner = basket_box(m, d)
    tx, ty = slot_xy(m, d, col)
    if verbose:
        print("任务：把%s [%.3f %.3f %.3f] 搬进篮子 [%.3f %.3f]，格位 [%.3f %.3f]（内腔 %.0f mm）"
              % (LABEL[col], *p0, bx, by, tx, ty, 2000 * inner))
        print()
        print("%-4s %-30s %-26s %-26s %s"
              % ("步", "动作", "方块位置", "指尖中点", "说明"))
    for i, item in enumerate(plan_for(col)):
        kind, keys, grip, note = item[0], item[1], item[2], item[3]
        hold = item[4] if len(item) > 4 else ()
        slow = item[5] if len(item) > 5 else ()
        if kind == "path":
            move_path(m, d, qad, keys, grip, viewer, hold=hold, slow=slow)
        else:
            dwell(m, d, qad, grip, viewer=viewer)
        p = d.xpos[bid].copy()
        pm = pad_mid(m, d)
        if verbose:
            print("%-4d %-30s [%.3f %.3f %.3f]        [%.3f %.3f %.3f]      %s"
                  % (i + 1, kind, *p, *pm, note))
    ok, p, bc, ft, inner = cube_in_basket(m, d, cube)
    if verbose:
        print()
        print("最终：%s [%.3f %.3f %.3f]" % (LABEL[col], *p))
        print("      自己的格位 (%.3f, %.3f)  底板顶面 z=%.3f  内半宽 %.3f"
              % (bc[0], bc[1], ft, inner))
        print("      偏离格位 %.1f mm（限 20.0），底面高差 %.1f mm（限 12.0）"
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
    """出过程图：沿连续轨迹跑到每个途经点时抓一张（轨迹本身不停车）。"""
    reset(m, d, qad)
    cube = CUBE_OF[col]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube)
    (bx, by), ft, inner = basket_box(m, d)
    tx, ty = slot_xy(m, d, col)
    # 机位按篮子真实位置算（篮子放大后又挪了位置，不能再写死坐标）
    marks = {
        "%s_grasp" % col: ("pp_%s_1_grasped" % col, (0.08, 0.02, 0.92), 0.95, 55, -14),
        "%s_lift" % col: ("pp_%s_2_lifted" % col, (0.04, 0.06, 1.00), 1.05, 55, -14),
        "%s_transit" % col: ("pp_%s_3_transit" % col,
                             ((0.08 + bx) / 2, (0.02 + by) / 2, 1.00), 1.10, 55, -16),
        "%s_place_hover" % col: ("pp_%s_4_above_basket" % col,
                                 (tx, ty, ft + 0.20), 0.95, 62, -18),
        # 篮子侧壁 90 mm 高、方块 80 mm 高 —— 低机位会被完全遮住，
        # 必须用俯视（el=-52）+ 近距离才能看到方块躺在篮底
        "%s_place" % col: ("pp_%s_5_placed" % col, (tx, ty, ft + 0.05), 0.62, 62, -52),
    }
    dt = m.opt.timestep
    shot = set()
    pending = []          # 本段结尾要拍的（等后续 dwell 结束后再拍）

    def take(name):
        if name in shot or name not in marks:
            return
        nm, look, dist, az, el = marks[name]
        shoot(m, d, nm, look, dist, az, el)
        shot.add(name)
        print("  写出 out/%s.png   %s z=%.4f" % (nm, LABEL[col], d.xpos[bid][2]))

    for item in plan_for(col):
        kind, keys, grip, note = item[0], item[1], item[2], item[3]
        hold = item[4] if len(item) > 4 else ()
        slow = item[5] if len(item) > 5 else ()
        if kind != "path":
            dwell(m, d, qad, grip)
            for name in pending:      # 夹爪动作结束才是"放好/夹住"的稳态
                take(name)
            pending = []
            continue
        Q = [np.array([d.qpos[a] for a in qad], dtype=float) if k is None
             else np.array([m.key(k).qpos[a] for a in qad], dtype=float)
             for k in keys]
        Q = np.array(Q)
        lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][0]
                       for j in ARM])
        hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][1]
                       for j in ARM])
        hold_i = {i for i, k in enumerate(keys) if k is not None and k in set(hold)}
        slow_i = set()
        for k in slow:
            if k in keys:
                i = keys.index(k)
                if i > 0:
                    slow_i.add(i - 1)
        samp, idx, T, _M = _build_path(Q, lo, hi, dt, hold=hold_i, slow=slow_i)
        n = len(samp)
        g0 = float(d.ctrl[7])
        ng = max(1, int(0.30 * n))
        # 中间途经点即时抓拍；最后一个途经点留到段末/夹爪动作之后
        want = {}
        for k in keys[1:-1]:
            if k in marks and k not in shot:
                want[idx[keys.index(k)]] = k
        for i in range(n + int(0.25 / dt)):
            if i < n:
                c = np.empty(m.nu)
                c[:7] = samp[i]
                c[7] = c[8] = (g0 + (grip - g0) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / ng))
                               if i < ng else grip)
                d.ctrl[:] = c
            else:
                d.ctrl[:7] = Q[-1]
                d.ctrl[7] = d.ctrl[8] = grip
            mujoco.mj_step(m, d)
            if i in want:
                take(want[i])
        if keys[-1] in marks and keys[-1] not in shot:
            pending.append(keys[-1])
    for name in pending:
        take(name)
    # 收尾全景 + 篮子俯视特写
    shoot(m, d, "pp_%s_6_done" % col, (bx + 0.10, by - 0.22, 0.88), 1.15, 60, -30)
    shoot(m, d, "pp_%s_7_basket_top" % col, (bx, by, ft + 0.05), 0.85, 62, -68)
    print("  写出 out/pp_%s_6_done.png / pp_%s_7_basket_top.png" % (col, col))
    ok, p, sc, ft, inner = cube_in_basket(m, d, cube)
    print()
    print("最终：%s [%.3f %.3f %.3f]" % (LABEL[col], *p))
    print("      偏离格位 (%.3f, %.3f) %.1f mm  底面高差 %.1f mm"
          % (sc[0], sc[1], 1000 * math.hypot(p[0] - sc[0], p[1] - sc[1]),
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
        placed = []
        worst = 0.0
        for col in cols:
            before = {}
            for c in placed:
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUBE_OF[c])
                before[c] = d.xpos[b].copy()
            run(m, d, qad, col, verbose=False)
            ok, p, sc, ft, inner = cube_in_basket(m, d, CUBE_OF[col])
            dxy = math.hypot(p[0] - sc[0], p[1] - sc[1])
            dz = abs(p[2] - (ft + 0.040))
            allok = allok and ok
            print("%-5s %s 落点 [%.3f %.3f %.3f]  偏离格位 [%.3f %.3f] %.1f mm（限 20.0）"
                  "  高差 %.1f mm（限 12.0）  -> %s"
                  % (col, LABEL[col], *p, sc[0], sc[1], 1000 * dxy,
                     1000 * dz, "PASS" if ok else "FAIL"))
            # 关键：此前已放好的方块必须原地不动（这是 `--all` 最初的真实 bug）
            for c, p0 in before.items():
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUBE_OF[c])
                dn = 1000 * math.hypot(d.xpos[b][0] - p0[0], d.xpos[b][1] - p0[1])
                worst = max(worst, dn)
                if dn > 5.0:
                    print("      !! 已放好的%s被挪动了 %.1f mm" % (LABEL[c], dn))
                    allok = False
            placed.append(col)
        if len(cols) > 1:
            print("\n搬运过程中已放置方块的最大水平扰动 %.2f mm（限 5.0）"
                  % worst)
            # 最终再整体校验一次：三块必须**同时**都在各自格位上
            print("同时检查三块是否都在篮内：")
            for col in COLORS:
                ok, p, sc, ft, inner = cube_in_basket(m, d, CUBE_OF[col])
                print("  %-5s [%.3f %.3f %.3f]  偏离格位 %.1f mm  -> %s"
                      % (col, *p, 1000 * math.hypot(p[0] - sc[0], p[1] - sc[1]),
                         "PASS" if ok else "FAIL"))
                allok = allok and ok
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
