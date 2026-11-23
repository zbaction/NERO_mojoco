#!/usr/bin/env python3
"""把 AgileX NERO 的 ROS URDF 精确转成 MuJoCo MJCF，并组装成"桌面 + 三色方块"场景。

数据来源（只读，不改动原仓库）
  E:\\postgraduate\\robotic_arm\\piper_ros\\src\\robot_description\\nero_description\\
      urdf\\nero_description.urdf   -> 关节链、origin xyz/rpy、轴向
      urdf\\nero_description.csv    -> 每连杆质量 / 质心 / 完整惯量张量
      meshes\\*.STL                 -> 已复制到 assets/nero/
  E:\\postgraduate\\robotic_arm\\NERO用户手册.html -> 关节限位、最大速度、夹爪参数（1.2 节）

转换要点
  * URDF rpy = Rz*Ry*Rx（定点旋转）-> MuJoCo quat，避免 euler 序列歧义
  * URDF joint origin 落到 child <body> 的 pos/quat；关节轴定义在 child 系，保持不变
  * URDF <inertia> 全张量 -> MJCF <inertial fullinertia="ixx iyy izz ixy ixz iyz">
    并用 CSV 的实测值覆盖，再按手册本体 4.8 kg 等比缩放（CSV 只是结构件质量）
  * 视觉用原厂 STL；碰撞用按网格包围盒手调的 capsule/cylinder/box
    （原 STL 单个 2~4 万三角面，直接做碰撞会慢到不可用）

用法：
    python3 build_nero.py       # -> nero_arm.xml, nero_table_scene.xml

关键帧位姿来自 poses.json（由 solve_poses.py 生成）；文件不存在时退回内置值。
"""
from __future__ import annotations

import json
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MESH_DIR = "assets/nero"
SRC_URDF = HERE / "assets" / "nero" / "nero_base.urdf"
SRC_CSV = HERE / "assets" / "nero" / "nero_description.csv"
POSES = HERE / "poses.json"

ARM_OUT = HERE / "nero_arm.xml"
SCENE_OUT = HERE / "nero_table_scene.xml"

# --- 用户手册 1.2 性能参数 ------------------------------------------------
ARM_BODY_KG = 4.75          # base_link..link7 合计；end_effector 自带 0.05 -> 本体 4.80
TABLE_Z = 0.74              # 桌面上表面高度（与 desk_three_cubes.xml 一致）
ARM_MOUNT = (-0.30, 0.0, TABLE_Z)
CUBE_HALF = 0.040           # 8 cm 方块（夹爪行程 100 mm，留 20 mm 接近余量）
CUBE_MASS = 0.30            # kg（塑料件量级；不是实心 1 kg）

# --- 篮子（放置目标，固定在桌面上，开口朝上）--------------------------------
# 内腔 180 mm 见方：8 cm 方块放进去后四周各留 50 mm，夹爪张开时也不会蹭到壁。
BASKET_XY = (-0.04, 0.42)   # 篮子中心（桌面坐标，台面 x∈[-0.4,0.4] y∈[-0.6,0.6]）
BASKET_INNER_HALF = 0.070   # 内腔半宽（内腔 140 mm 见方）
BASKET_WALL = 0.008         # 侧壁厚
BASKET_FLOOR = 0.008        # 底板厚
BASKET_HEIGHT = 0.090       # 侧壁高（从底板上表面算起）
# 放置时方块中心的高度：底板顶面 + 半个方块
PLACE_Z = TABLE_Z + BASKET_FLOOR + CUBE_HALF
PLACE_HOVER_DZ = 0.120      # 放置前悬停高度

JOINT_LIMIT_DEG = {
    "joint1": (-157.0, 157.0), "joint2": (-102.0, 102.0), "joint3": (-160.0, 160.0),
    "joint4": (-60.0, 125.0), "joint5": (-160.0, 160.0), "joint6": (-44.0, 57.0),
    "joint7": (-97.0, 97.0),
}
# 手册：J1–J3 180°/s，J4–J7 225°/s
JOINT_VEL = {n: (math.pi if i < 3 else 1.25 * math.pi) for i, n in
             enumerate("joint%d" % i for i in range(1, 8))}
JOINT_TORQUE_MAX = 24.0     # 峰值 N*m（手册 MIT 模式 t_ff ±16 N*m，留余量）

# --- 夹爪（手册：行程 0~100 mm、额定夹合力 40 N、最大 50 N、本体 500 g）----
GRIPPER_MASS = 0.5
# 坐标系：gripper_base 原点在法兰外 30 mm，+z 沿工具轴向外（工具朝下时即向下）。
#   手掌   z 0.000~0.040
#   指体   z 0.040~0.115（75 mm 长）
#   指尖   z 0.085~0.115（pad 中心 z=0.100）
# 夹爪闭合时两 pad 内表面正好接触（开度 0）；ctrl=GRIP_OPEN -> 开度 100 mm。
FINGER_STROKE = 0.055       # 单指行程 55 mm -> 最大开度 110 mm
FINGER_HALF_GAP = 0.014     # 指体中心到工具轴的距离（闭合时两指相贴）
PAD_HALF_Y = 0.006
PAD_OFFSET = 0.130          # 法兰原点到 pad 中心沿工具轴的距离（=0.030+0.100）
GRIPPER_FORCE = 40.0        # N，额定夹合力（执行器上限 50 N）
GRIP_KP, GRIP_KV = 3000.0, 6.0
GRIP_OPEN = 0.050           # 接近/张开时的 ctrl（开度 100 mm）
GRIP_CLOSE = 0.0            # 闭合 ctrl；位置伺服压到方块上由 forcerange 限在 50 N

# --- 每个连杆的碰撞体：类型 / 中心 / 尺寸 / 绕 x 轴转角（贴合 STL 包围盒）--
COLLISION = {
    "base_link":    ("cylinder", (0.0, 0.0, 0.0430), (0.0500, 0.0430), 0.0),
    "link1":        ("cylinder", (0.0, 0.0, -0.0135), (0.0450, 0.0435), 0.0),
    "link2":        ("capsule",  (0.0, -0.1000, 0.0), (0.0400, 0.0900), 90.0),
    "link3":        ("capsule",  (0.0, 0.0, -0.0850), (0.0350, 0.0850), 0.0),
    "link4":        ("capsule",  (0.0, -0.0350, 0.0), (0.0340, 0.0350), 90.0),
    "link5":        ("capsule",  (0.0, 0.0, -0.0725), (0.0350, 0.0675), 0.0),
    "link6":        ("sphere",   (0.0, 0.0, 0.0), (0.0260,), 0.0),
    "link7":        ("cylinder", (0.0, 0.0, -0.0048), (0.0140, 0.0048), 0.0),
    "end_effector": ("box",      (0.0, 0.0, -0.0180), (0.0210, 0.0275, 0.0300), 0.0),
}

MESH_OF = {
    "base_link": "base_link.STL", "link1": "Link1.STL", "link2": "Link2.STL",
    "link3": "Link3.STL", "link4": "Link4.STL", "link5": "Link5.STL",
    "link6": "Link6.STL", "link7": "Link7.STL", "end_effector": "stand_v2.STL",
}
MAT_OF = {"base_link": "nero_dark", "link6": "nero_white", "link7": "nero_white",
          "end_effector": "nero_white"}

# position 伺服增益：越靠近基座负载越大
GAINS = {"joint1": (400.0, 12.0), "joint2": (400.0, 12.0), "joint3": (300.0, 9.0),
         "joint4": (200.0, 6.0), "joint5": (120.0, 4.0), "joint6": (50.0, 2.0),
         "joint7": (25.0, 1.0)}

# 碰撞分组掩码：1=静态环境  2=机械臂  4=可操作物体
ARM_CONTYPE, ARM_CONAFF = 2, (1 | 4)
OBJ_CONTYPE, OBJ_CONAFF = 4, (1 | 2 | 4)

# 关键帧（度），由 solve_poses.py 求解；此处为找不到 poses.json 时的兜底
FALLBACK_POSES = {
    "ready": [-6.62, -47.14, 46.07, -25.65, 54.96, -23.14, 80.81],
    "grasp": [36.78, -76.80, 99.58, 73.02, -15.48, 5.06, -14.39],
}


# ------------------------------------------------------------------ 工具
def rpy_to_quat(roll, pitch, yaw):
    """URDF 的 rpy = Rz(yaw)*Ry(pitch)*Rx(roll) -> (w, x, y, z)。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def f(v, nd=6):
    s = ("%.*f" % (nd, float(v))).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "", "-0.0") else s


def vec(vals, nd=6):
    return " ".join(f(v, nd) for v in vals)


def camera_xml(name, pos, lookat, fovy):
    """用 pos + 看向点生成 MJCF camera（xyaxes = 右 / 上）。

    视线接近竖直时要换一个参考"上方向"，否则 cross 退化。
    """
    p = np.asarray(pos, float)
    fwd = np.asarray(lookat, float) - p
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up_w))) > 0.999:
        up_w = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return ('    <camera name="%s" pos="%s" xyaxes="%s %s" fovy="%s"/>'
            % (name, vec(p, 3), vec(right, 4), vec(up, 4), f(fovy, 1)))


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, []
    for el in root.findall("link"):
        ine = el.find("inertial")
        d = {"mass": 0.0, "com": (0.0, 0.0, 0.0), "I": (0.0,) * 6}
        if ine is not None:
            d["mass"] = float(ine.find("mass").get("value"))
            o = ine.find("origin")
            if o is not None and o.get("xyz"):
                d["com"] = tuple(float(x) for x in o.get("xyz").split())
            i = ine.find("inertia")
            d["I"] = tuple(float(i.get(k)) for k in
                           ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        links[el.get("name")] = d
    for el in root.findall("joint"):
        o, ax = el.find("origin"), el.find("axis")
        joints.append({
            "name": el.get("name"), "type": el.get("type"),
            "parent": el.find("parent").get("link"),
            "child": el.find("child").get("link"),
            "xyz": tuple(float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()),
            "rpy": tuple(float(x) for x in (o.get("rpy") if o is not None else "0 0 0").split()),
            "axis": tuple(float(x) for x in (ax.get("xyz") if ax is not None else "0 0 1").split()),
        })
    return links, joints


def chain_from(joints, root="base_link"):
    out, cur, seen = [], root, set()
    while True:
        nxt = [j for j in joints if j["parent"] == cur and j["name"] not in seen]
        if not nxt:
            return out
        j = nxt[0]
        seen.add(j["name"])
        out.append(j)
        cur = j["child"]


ARM_LINKS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7"]


def apply_csv_inertia(links, csv_path):
    """用 nero_description.csv 的实测质量/质心/惯量替换 URDF 里的占位值。

    CSV 的惯量列序是 Ixx, Ixy, Ixz, Iyy, Iyz, Izz，
    与 MJCF fullinertia 要求的 Ixx, Iyy, Izz, Ixy, Ixz, Iyz 不同，必须重排。
    """
    import csv as _csv
    rows = list(_csv.DictReader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    by_name = {r["Link Name"].strip().lower(): r for r in rows}
    applied = 0
    for name, d in links.items():
        r = by_name.get(name.lower())
        if r is None:
            continue
        d["mass"] = float(r["Mass"])
        d["com"] = (float(r["Center of Mass X"]), float(r["Center of Mass Y"]),
                    float(r["Center of Mass Z"]))
        d["I"] = (float(r["Moment Ixx"]), float(r["Moment Iyy"]), float(r["Moment Izz"]),
                  float(r["Moment Ixy"]), float(r["Moment Ixz"]), float(r["Moment Iyz"]))
        applied += 1
    return applied


def scale_arm_mass(links, target_kg):
    """CSV 的连杆质量是 SolidWorks 结构件质量，不含 7 个电机/减速器，合计仅 1.95 kg，
    手册 1.2 给的本体是 4.8 kg。等比放大到手册值；几何不变 -> I 同比例缩放，
    质心位置保持不动。
    """
    cur = sum(links[n]["mass"] for n in ARM_LINKS)
    if cur <= 0:
        return 1.0
    s = target_kg / cur
    for n in ARM_LINKS:
        links[n]["mass"] *= s
        links[n]["I"] = tuple(v * s for v in links[n]["I"])
    return s


def inertial_xml(d, ind):
    return ('%s<inertial pos="%s" mass="%s" fullinertia="%s"/>'
            % (ind, vec(d["com"], 9), f(d["mass"], 9), vec(d["I"], 12)))


def collision_xml(name, ind):
    ctype, pos, size, rotx = COLLISION[name]
    a = ['name="%s_col"' % name, 'type="%s"' % ctype, 'pos="%s"' % vec(pos, 5)]
    if ctype in ("capsule", "cylinder"):
        a.append('size="%s %s"' % (f(size[0], 5), f(size[1], 5)))
    elif ctype == "sphere":
        a.append('size="%s"' % f(size[0], 5))
    else:
        a.append('size="%s"' % vec(size, 5))
    if rotx:
        a.append('quat="%s"' % vec(rpy_to_quat(math.radians(rotx), 0, 0)))
    a += ['contype="%d"' % ARM_CONTYPE, 'conaffinity="%d"' % ARM_CONAFF,
          'condim="3"', 'friction="1.0 0.02 0.002"',
          'solref="0.005 1"', 'solimp="0.95 0.99 0.001"']
    return ind + "<geom " + " ".join(a) + "/>"


LINKS = {}


def gripper_xml(ind):
    """两指平行夹爪（手册：行程 0~100 mm、夹合力 40 N、本体 500 g）。

    gripper_base 原点在法兰外 30 mm 处，+z 沿工具轴向外（工具朝下时即向下）。
      手掌  z 0.000~0.040
      指体  z 0.040~0.115（指体中心 z=0.0775）
      指尖  z 0.085~0.115（pad 中心 z=0.100，即法兰原点沿工具轴 0.130 处）
    指体在 ±FINGER_HALF_GAP，沿 ±y 滑出张开；ctrl=0 时两 pad 内表面接触，
    ctrl=GRIP_OPEN 时开度 100 mm，正好等于方块边长。
    """
    g = []
    g.append('%s<inertial pos="0 0 0.040" mass="%s" fullinertia="0.0004 0.0005 0.0004 0 0 0"/>'
             % (ind, f(GRIPPER_MASS * 0.55, 6)))
    g.append('%s<geom name="gripper_base_vis" type="box" pos="0 0 0.020" size="0.032 0.052 0.020" '
             'material="nero_dark" contype="0" conaffinity="0" group="2"/>' % ind)
    # 手掌碰撞：只到 z=0.040，比指尖高 75 mm，不会碰到方块顶面
    g.append('%s<geom name="gripper_base_col" type="box" pos="0 0 0.020" size="0.030 0.050 0.020" '
             'contype="%d" conaffinity="%d" condim="3" friction="0.8 0.02 0.002" '
             'solref="0.005 1" solimp="0.95 0.99 0.001"/>' % (ind, ARM_CONTYPE, ARM_CONAFF))

    for side, sgn in (("l", 1.0), ("r", -1.0)):
        name = "finger_%s" % side
        g.append('%s<body name="%s" pos="0 %s 0">'
                 % (ind, name, f(sgn * FINGER_HALF_GAP, 4)))
        g.append('%s  <joint name="grip_%s" type="slide" axis="0 %s 0" range="0 %s" '
                 'limited="true" armature="0.002" damping="3.0" frictionloss="0.05" '
                 'actuatorfrcrange="-%s %s"/>'
                 % (ind, side, "1" if sgn > 0 else "-1", f(FINGER_STROKE, 4),
                    f(50.0, 2), f(50.0, 2)))
        g.append('%s  <inertial pos="0 0 0.078" mass="%s" '
                 'fullinertia="1.2e-5 1.6e-5 1.2e-5 0 0 0"/>'
                 % (ind, f(GRIPPER_MASS * 0.225, 6)))
        # 指体：z 0.040~0.115，仅视觉（碰撞完全交给 pad）
        g.append('%s  <geom name="%s_vis" type="box" pos="0 0 0.0775" size="0.010 0.007 0.0375" '
                 'material="nero_white" contype="0" conaffinity="0" group="2"/>'
                 % (ind, name))
        # 指尖摩擦块：内表面在 ctrl=0 时正好落在工具轴上（两指相贴）
        g.append('%s  <geom name="%s_pad" type="box" pos="0 %s 0.100" size="0.010 %s 0.015" '
                 'contype="%d" conaffinity="%d" condim="4" friction="2.4 0.06 0.0005" '
                 'solref="0.003 1" solimp="0.98 0.998 0.001"/>'
                 % (ind, name, f(-sgn * (FINGER_HALF_GAP - PAD_HALF_Y), 4),
                    f(PAD_HALF_Y, 4), ARM_CONTYPE, ARM_CONAFF))
        g.append('%s</body>' % ind)
    return g


def joint_xml(j, ind):
    if j["type"] != "revolute":
        return []       # fixed 关节的变换已写进 body pos/quat，不再单独建 joint
    lo, hi = (math.radians(x) for x in JOINT_LIMIT_DEG[j["name"]])
    n = int(j["name"][-1])
    return ['%s<joint name="%s" type="hinge" pos="0 0 0" axis="%s"'
            % (ind, j["name"], vec(j["axis"])),
            '%s       range="%s %s" limited="true" armature="%s"'
            % (ind, f(lo), f(hi), f(0.010 + 0.004 * (7 - n))),
            '%s       damping="%s" frictionloss="0.05"'
            % (ind, f(0.40 + 0.15 * (7 - n))),
            '%s       actuatorfrcrange="%s %s"/>'
            % (ind, f(-JOINT_TORQUE_MAX), f(JOINT_TORQUE_MAX))]


def basket_xml(ind="    "):
    """桌面上的收纳篮（固定件）：底板 + 四壁，开口朝上。

    内腔 140 mm 见方、深 90 mm —— 8 cm 方块放进去四周各留 30 mm，
    夹爪下降到腔体里张开/退出都不会蹭壁。
    """
    ih = BASKET_INNER_HALF
    oh = ih + BASKET_WALL                       # 外半宽
    t = BASKET_FLOOR                            # 底板厚
    cz = t + BASKET_HEIGHT / 2.0                # 侧壁中心高
    hz = BASKET_HEIGHT / 2.0                    # 侧壁半高
    x, y = BASKET_XY
    ct, ca = 1, 7                               # 与桌子同组：会挡臂、也挡方块
    geo = ('type="box" material="mat_basket" contype="%d" conaffinity="%d" '
           'condim="3" friction="0.8 0.02 0.002" solref="0.004 1" '
           'solimp="0.95 0.99 0.001"' % (ct, ca))
    L = []
    L.append('%s<body name="basket" pos="%s %s %s">'
             % (ind, f(x, 3), f(y, 3), f(TABLE_Z, 3)))
    L.append('%s  <geom name="basket_floor" %s pos="0 0 %s" size="%s %s %s"/>'
             % (ind, geo, f(t / 2, 4), f(oh, 3), f(oh, 3), f(t / 2, 4)))
    # 沿 y 方向的两面长壁（在 ±x 处），长度取内腔，避免与另外两面重叠
    for sx in (-1, 1):
        L.append('%s  <geom name="basket_wall_x%s" %s pos="%s 0 %s" '
                 'size="%s %s %s"/>'
                 % (ind, "p" if sx > 0 else "m", geo,
                    f(sx * (ih + BASKET_WALL / 2), 4), f(cz, 4),
                    f(BASKET_WALL / 2, 4), f(ih, 3), f(hz, 4)))
    # 沿 x 方向的两面短壁（在 ±y 处），长度取外半宽
    for sy in (-1, 1):
        L.append('%s  <geom name="basket_wall_y%s" %s pos="0 %s %s" '
                 'size="%s %s %s"/>'
                 % (ind, "p" if sy > 0 else "m", geo,
                    f(sy * (ih + BASKET_WALL / 2), 4), f(cz, 4),
                    f(oh, 3), f(BASKET_WALL / 2, 4), f(hz, 4)))
    L.append('%s</body>' % ind)
    return "\n".join(L)


def body_xml(link, parent_joint, joints, ind, depth=0):
    """输出 link 及其下游子树。

    URDF 的 joint origin (xyz, rpy) 在 MJCF 里落到 child <body> 的 pos/quat；
    关节轴在 URDF 中本来定义于 child 连杆系，因此 axis 原样保留。
    gravcomp="1"：重力前馈（等价于真实控制器里的重力补偿项），
    否则纯 P 伺服在重力下会有 1~1.5° 的稳态误差。
    """
    d = LINKS[link]
    pj = parent_joint
    pos = vec(pj["xyz"]) if pj else "0 0 0"
    quat = vec(rpy_to_quat(*pj["rpy"])) if pj else "1 0 0 0"
    L = [ind + '<body name="%s" pos="%s" quat="%s" gravcomp="1">'
         % ("nero_base" if depth == 0 else link, pos, quat)]
    if pj is not None:
        L.extend(joint_xml(pj, ind + "  "))
    L.append(inertial_xml(d, ind + "  "))
    if link in MESH_OF:
        L.append('%s<geom name="%s_vis" type="mesh" mesh="nero_%s_vis" material="%s" '
                 'contype="%d" conaffinity="%d" group="2"/>'
                 % (ind + "  ", link, link, MAT_OF.get(link, "nero_light"),
                    ARM_CONTYPE, ARM_CONAFF))
        L.append(collision_xml(link, ind + "  "))

    if link == "end_effector":
        L.append('%s  <body name="gripper_base" pos="0 0 0.030">' % ind)
        L.extend(gripper_xml(ind + "    "))
        L.append('%s  </body>' % ind)

    child = next((j for j in joints if j["parent"] == link), None)
    if child is not None:
        L.extend(body_xml(child["child"], child, joints, ind + "  ", depth + 1))
    L.append(ind + "</body>")
    return L


def load_poses():
    if POSES.exists():
        return json.loads(POSES.read_text(encoding="utf-8"))
    return FALLBACK_POSES


# ------------------------------------------------------------------ 主流程
def main():
    global LINKS
    LINKS, joints = parse_urdf(SRC_URDF)
    n_csv = apply_csv_inertia(LINKS, SRC_CSV)
    mscale = scale_arm_mass(LINKS, ARM_BODY_KG)
    chain = chain_from(joints)
    names = [j["name"] for j in chain]
    assert names[:7] == ["joint%d" % i for i in range(1, 8)], names
    assert names[7:] == ["end_effector_joint"], names

    meshes = "\n".join('    <mesh name="nero_%s_vis" file="%s/%s"/>' % (k, MESH_DIR, v)
                       for k, v in MESH_OF.items())
    mats = """    <material name="nero_light" rgba="0.78 0.80 0.83 1" specular="0.35" shininess="0.55" reflectance="0.04"/>
    <material name="nero_dark"  rgba="0.235 0.248 0.275 1" specular="0.30" shininess="0.55"/>
    <material name="nero_white" rgba="0.72 0.735 0.755 1" specular="0.35" shininess="0.60"/>"""
    acts = "\n".join(
        '    <position name="%s_act" joint="%s" kp="%s" kv="%s" ctrlrange="%s %s" forcerange="%s %s"/>'
        % (n, n, f(GAINS[n][0]), f(GAINS[n][1]),
           f(math.radians(JOINT_LIMIT_DEG[n][0])), f(math.radians(JOINT_LIMIT_DEG[n][1])),
           f(-JOINT_TORQUE_MAX), f(JOINT_TORQUE_MAX)) for n in names[:7])
    acts += "\n"
    for side in ("l", "r"):
        # 注意 forcerange 必须双向：闭合时需要负向力把手指收回来
        acts += ('    <position name="grip_%s_act" joint="grip_%s" kp="%s" kv="%s" '
                 'ctrlrange="0 %s" forcerange="-%s %s"/>\n'
                 % (side, side, f(GRIP_KP), f(GRIP_KV),
                    f(FINGER_STROKE, 4), f(50.0, 2), f(50.0, 2)))

    arm_body = "\n".join(body_xml("base_link", None, joints, "    "))

    lighting = """    <light name="key"  pos="1.10 -1.20 2.30" dir="-0.45 0.50 -1" directional="true"
           diffuse="0.42 0.42 0.42" specular="0.18 0.18 0.18" castshadow="true"/>
    <light name="fill" pos="-1.20 1.10 1.80" dir="0.55 -0.50 -1" directional="true"
           diffuse="0.22 0.23 0.26" specular="0.06 0.06 0.06"/>
    <light name="rim"  pos="-0.60 -1.60 1.60" dir="0.30 0.80 -0.75" directional="true"
           diffuse="0.18 0.19 0.22"/>"""

    common_head = """  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720" fovy="42"/>
    <headlight ambient="0.26 0.27 0.30" diffuse="0.30 0.30 0.31" specular="0.06 0.06 0.06"/>
    <map znear="0.01" zfar="30"/>
    <quality shadowsize="4096"/>
    <rgba haze="0.20 0.22 0.25 1"/>
  </visual>"""

    # <statistic> 决定 MuJoCo 默认 free 相机的取景中心与距离。
    # 不写的话默认 center=(0,0,0)、extent=2 —— 视线落在世界原点（地面），
    # 桌面上的机械臂会被顶出画面上边缘（实测 96.7% 顶点出画），
    # 交互窗口里看起来就像"只显示了一部分机械臂"。
    stat_arm = ('  <statistic center="0 0 0.42" extent="1.10"/>')
    stat_scene = ('  <statistic center="0.00 0.00 0.80" extent="2.10"/>')

    poses = load_poses()
    kf_names = [k for k in ("ready", "hover", "grasp", "grasp_red", "grasp_blue",
                            "lift", "transit", "place_hover", "place")
                if k in poses]
    # 每个关键帧都写全 qpos（7 关节 + 2 指）。所有位姿指尖都保持张开（GRIP_OPEN，
    # 开度 100 mm），闭合由仿真脚本把 ctrl 拉到 GRIP_CLOSE 完成。
    kf_xml = "\n".join(
        '    <key name="%s" qpos="%s %s %s"/>'
        % (k, " ".join(f(math.radians(v), 5) for v in poses[k]),
           f(GRIP_OPEN, 4), f(GRIP_OPEN, 4)) for k in kf_names)

    # ---------------- 1) 独立机械臂模型（基座原点，朝 +z）------------------
    ARM_OUT.write_text("""<mujoco model="agilex_nero_arm">
  <!-- 由 build_nero.py 从 AgileX 官方 URDF + CSV 自动生成，勿手改 -->
""" + common_head + "\n" + stat_arm + """

  <asset>
""" + mats + "\n" + meshes + """
  </asset>

  <worldbody>
""" + lighting + """
    <geom name="ground" type="plane" size="0 0 0.05" rgba="0.20 0.22 0.25 1"
          contype="1" conaffinity="7"/>
""" + arm_body + """
""" + camera_xml("arm_iso", (1.05, -1.05, 1.30), (0.0, 0.0, 0.40), 42) + """
""" + camera_xml("arm_front", (0.02, -1.35, 0.55), (0.0, 0.0, 0.40), 42) + """
  </worldbody>

  <actuator>
""" + acts + """  </actuator>

  <keyframe>
""" + kf_xml + """
  </keyframe>
</mujoco>
""", encoding="utf-8")

    # ---------------- 2) 桌面场景：桌子 + 三个方块 + 机械臂 --------------
    def qdeg(vals):
        return " ".join(f(math.radians(v), 5) for v in vals)

    cubes = [("cube_red", (0.16, -0.26), 12.0, "mat_red"),
             ("cube_green", (0.18, 0.00), -8.0, "mat_green"),
             ("cube_blue", (0.15, 0.26), 20.0, "mat_blue")]
    cube_xml = "\n".join(
        """    <body name="{n}" pos="{x} {y} {z}" quat="{q}">
      <freejoint name="{n}_free"/>
      <geom name="{n}_geom" type="box" size="{s} {s} {s}" material="{m}" mass="{mass}"
            friction="1.0 0.02 0.002" contype="{ct}" conaffinity="{ca}"/>
    </body>""".format(n=n, x=f(x, 3), y=f(y, 3), z=f(TABLE_Z + CUBE_HALF, 3),
                      q=vec(rpy_to_quat(0.0, 0.0, math.radians(yaw))),
                      m=m, mass=f(CUBE_MASS, 3), s=f(CUBE_HALF, 3),
                      ct=OBJ_CONTYPE, ca=OBJ_CONAFF)
        for n, (x, y), yaw, m in cubes)

    mount_body = "\n".join(body_xml("base_link", None, joints, "      "))

    SCENE_OUT.write_text("""<mujoco model="nero_desk_scene">
  <!-- 桌面场景：AgileX NERO 七自由度机械臂 + 桌面三个彩色方块
       由 build_nero.py 生成；机械臂基座固定在桌面上表面 (z=0.74) -->
""" + common_head + "\n" + stat_scene + """

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.34 0.38 0.44" rgb2="0.05 0.06 0.08" width="512" height="512"/>
    <texture name="floor_grid" type="2d" builtin="checker"
             rgb1="0.28 0.30 0.33" rgb2="0.22 0.24 0.27" width="512" height="512"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="8 8" reflectance="0.03"/>
    <material name="wood"      rgba="0.66 0.46 0.30 1" specular="0.25" shininess="0.25" reflectance="0.02"/>
    <material name="leg_mat"   rgba="0.30 0.33 0.38 1" specular="0.4" shininess="0.5"/>
    <material name="mat_red"   rgba="0.86 0.13 0.13 1" specular="0.3" shininess="0.6"/>
    <material name="mat_green" rgba="0.12 0.68 0.26 1" specular="0.3" shininess="0.6"/>
    <material name="mat_blue"  rgba="0.16 0.34 0.86 1" specular="0.3" shininess="0.6"/>
    <material name="mat_basket" rgba="0.55 0.40 0.22 1" specular="0.15" shininess="0.2"/>
""" + mats + "\n" + meshes + """
  </asset>

  <worldbody>
""" + lighting + """

    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"
          contype="1" conaffinity="7"/>

    <!-- 桌子：桌面 0.8m(x) x 1.2m(y)，台面高 0.74m -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" pos="0 0 0.72" size="0.40 0.60 0.02"
            material="wood" friction="0.9 0.02 0.002" contype="1" conaffinity="7"/>
      <geom name="leg_0" type="box" pos=" 0.355  0.555 0.35" size="0.022 0.022 0.35" material="leg_mat" contype="1" conaffinity="7"/>
      <geom name="leg_1" type="box" pos="-0.355  0.555 0.35" size="0.022 0.022 0.35" material="leg_mat" contype="1" conaffinity="7"/>
      <geom name="leg_2" type="box" pos=" 0.355 -0.555 0.35" size="0.022 0.022 0.35" material="leg_mat" contype="1" conaffinity="7"/>
      <geom name="leg_3" type="box" pos="-0.355 -0.555 0.35" size="0.022 0.022 0.35" material="leg_mat" contype="1" conaffinity="7"/>
      <geom name="rail_x" type="box" pos="0 0.555 0.10" size="0.355 0.012 0.012" material="leg_mat" contype="1" conaffinity="7"/>
      <geom name="rail_y" type="box" pos="0.355 0 0.10" size="0.012 0.555 0.012" material="leg_mat" contype="1" conaffinity="7"/>
    </body>

    <!-- AgileX NERO 机械臂，基座用 4x M5 固定在桌面上 -->
    <body name="nero_mount" pos=\"""" + vec(ARM_MOUNT, 3) + """\">
""" + mount_body + """
    </body>

    <!-- 三个方块：边长 8 cm，摆在台面上 -->
""" + cube_xml + """

    <!-- 收纳篮：放置任务的目标容器 -->
""" + basket_xml() + """

""" + camera_xml("overview", (1.05, -1.20, 1.42), (-0.10, 0.05, 0.92), 40) + """
""" + camera_xml("arm_front", (-0.28, -0.92, 1.10), (-0.10, 0.02, 1.00), 42) + """
""" + camera_xml("arm_side", (0.62, -0.60, 1.18), (-0.16, 0.02, 1.00), 42) + """
""" + camera_xml("top", (0.05, 0.00, 2.10), (0.05, 0.00, 0.80), 44) + """
  </worldbody>

  <actuator>
""" + acts + """  </actuator>

  <keyframe>
""" + kf_xml + """
  </keyframe>
</mujoco>
""", encoding="utf-8")

    total = sum(LINKS[n]["mass"] for n in LINKS) + GRIPPER_MASS
    print("写出 %s （%.1f KB）" % (ARM_OUT.name, ARM_OUT.stat().st_size / 1024))
    print("写出 %s （%.1f KB）" % (SCENE_OUT.name, SCENE_OUT.stat().st_size / 1024))
    print("关节链: " + " -> ".join(names))
    print("连杆 %d 个（%d 个用 CSV 实测惯量，结构件质量缩放 %.3f）" % (len(LINKS), n_csv, mscale))
    print("机械臂 + 夹爪总质量 %.3f kg（本体 4.80 kg + 夹爪 0.50 kg）" % total)
    print("安装位置 x=%.2f y=%.2f z=%.2f（桌面 z=%.2f）" % (*ARM_MOUNT, TABLE_Z))
    print("关键帧: " + ", ".join(kf_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
