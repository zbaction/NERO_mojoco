# NERO 机械臂 MuJoCo 场景

AgileX **NERO** 七自由度机械臂 + 两指夹爪，装在木桌上，桌面上摆着红/绿/蓝三个方块，
旁边放着一个木色收纳篮。模型由官方 URDF/质量表逐项转换为 MuJoCo MJCF，
机械臂可以真的走过去**夹起方块、搬过去、放进篮子里**。

| | |
| --- | --- |
| MuJoCo | 3.17.0 |
| 解释器 | `/home/zhangb/so101_mujoco/.venv/bin/python`（系统 python3 没有 mujoco） |
| WSL 路径 | `~/NERO` |
| Windows 路径 | `\\wsl.localhost\Ubuntu-22.04\home\zhangb\NERO` |

## 快速开始

```bash
cd ~/NERO
./run.sh              # 交互窗口（WSLg 显示到 Windows 桌面）
./run.sh demo         # 交互窗口，自动演示：张开→下降→夹紧→提起
./run.sh pick         # 交互窗口，完整演示：夹起绿方块 → 搬进篮子 → 松手退出
./run.sh pick red     # 同上，但搬红方块（也支持 green / blue / all）
./run.sh check        # 无窗口自检，抓取成功返回 0
./run.sh place        # 无窗口自检：绿方块落进篮子返回 0
./run.sh place blue   # 同上，但搬蓝方块（也支持 red / all）
./run.sh validate     # 6 段全量校验
./run.sh shots        # 重出成果图到 out/
./run.sh pp-shots all # 重出三色各 7 张过程图到 out/
./run.sh detail       # 夹爪特写图到 out/
./run.sh desk         # 只看第一版的桌子 + 三色方块
```

交互窗口里鼠标左键旋转、右键平移、滚轮缩放，`Tab` 切换预设相机，
空格暂停，`Esc` 退出。

## 场景内容

- **桌子**：桌面 0.8 m × 1.2 m，上表面 z = 0.740 m，四条腿 + 横撑。
- **机械臂**：AgileX NERO，7 个转动关节 + 末端两指夹爪，基座装在桌面
  `(-0.30, 0.0, 0.74)`，用一块 0.10 m × 0.10 m × 0.014 m 的转接板固定。
- **方块**：边长 **0.080 m**（8 cm），质量各 0.30 kg，带 `freejoint` 参与真实物理，
  初始绕 z 轴各有一点偏转：红 +12°、绿 -8°、蓝 +20°，位置分别为
  `(0.16, -0.26)`、`(0.18, 0)`、`(0.15, +0.26)`。
- **收纳篮**：中心 `(-0.04, 0.42)`，外廓 0.156 m 见方、侧壁高 0.090 m、壁厚 8 mm，
  内腔 **140 mm × 140 mm**，底板顶面 z = 0.748 m。5 个 box 拼成（底板 + 四面侧壁），
  也是真实碰撞体，方块扔进去会真的撞在壁上、落在底板上。

## 模型数据来源

| 内容 | 来源 |
| --- | --- |
| 关节链 / 关节原点 / 连杆网格 | `E:\postgraduate\robotic_arm\piper_ros\src\robot_description\nero_description\urdf\nero_description.urdf` |
| 质量、质心、惯量张量 | 同目录 `nero_description.csv` |
| 关节限位、速度、负载、臂展 | `E:\postgraduate\robotic_arm\NERO用户手册.html` 1.2 节 |

`E:\postgraduate\robotic_arm\assets\NERO 模型文件` 里只有 STEP 格式的 CAD 整机，
没有 URDF/mesh，所以实际几何取自上面的 ROS 描述包。

网格文件在 `assets/nero/`：`Link1..Link8.STL`、`base_link.STL`、`gripper.STL`、`stand_v2.STL`。

### 转换时做的处理

1. **URDF → MJCF 手工重写**，而不是让 MuJoCo 直接读 URDF
   （`package://` 路径解析失败、`.dae` 缺失、URDF 里的惯量是陈旧占位值）。
2. **质量缩放**：CSV 里的结构件合计只有 1.947 kg（不含电机/减速器），
   按手册本体 4.8 kg 等比放大 2.531 倍，惯量同比缩放、质心不动。
   最终本体（含法兰）**4.800 kg**、夹爪 **0.500 kg**。
3. **惯量张量列序重排**：CSV 是 `Ixx,Ixy,Ixz,Iyy,Iyz,Izz`，
   MJCF `fullinertia` 要 `Ixx,Iyy,Izz,Ixy,Ixz,Iyz`。
4. **碰撞体手调简化**：视觉用 STL 网格，碰撞用 capsule / cylinder / sphere / box
   近似（见 `build_nero.py` 的 `COLLISION` 表），避免网格碰撞拖慢仿真。
5. **关节类型**：URDF 的 `revolute` → MJCF 的 `hinge`（MuJoCo 没有 `revolute` 这个关键字）。

## 关键实现细节

### 关节限位（与手册 1.2 节逐项一致）

| 关节 | 范围 (°) | 最大速度 |
| --- | --- | --- |
| J1 | -157 ~ 157 | 180 °/s |
| J2 | -102 ~ 102 | 180 °/s |
| J3 | -160 ~ 160 | 180 °/s |
| J4 | -60 ~ 125 | 225 °/s |
| J5 | -160 ~ 160 | 225 °/s |
| J6 | -44 ~ 57 | 225 °/s |
| J7 | -97 ~ 97 | 225 °/s |

### 夹爪

- 行程 0 ~ 110 mm（`FINGER_STROKE = 0.055`，每指 55 mm）。
- `qpos = 0` 是**闭合**，`qpos = 0.055` 是**全开**；两指内间隙 = `2 × qpos`。
- 指尖 pad 用 `condim=4`、`friction="2.4 0.06 0.0005"`，
  `solref="0.003 1"`、`solimp="0.98 0.998 0.001"`，保证夹持不打滑。
- 执行器力双向（`forcerange="-50 50"`）——只给正向力的话手指合不拢。

### 位姿求解（`solve_poses.py`）

关节 J7 限位只有 ±97°，完整 6 自由度姿态在桌面高度经常不可达。实测：

- 只约束位置：位置误差 0 mm，但工具轴倾斜 **56.5°**（夹爪侧着）。
- 约束位置 + 工具轴朝下（5 DOF）：**位置误差 0.00 mm、倾角 0.00°** ← 采用。
- 约束完整姿态（6 DOF）：位置误差 15~65 mm，J7 顶到限位。

最终用 **6 组残差解 7 个关节**（欠定，靠阻尼最小二乘取最小范数解）：
位置 3 + 工具轴倾角 3 + **夹持面偏航 1**。夹持面偏航这一步是关键——
两指连线必须对齐世界 ±y 轴，否则指尖会从侧面撞上方块而不是从上方夹进去。

`poses.json` 里的 9 个位姿全部 **位置误差 0.00 mm、倾角 0.00°、偏航误差 0.00°**：

| 关键帧 | 关节角 (°) | 余量 |
| --- | --- | --- |
| ready | 19.93, -54.13, 83.01, 34.64, -17.40, -32.56, -22.10 | 11.4° |
| hover | 34.22, -68.41, 88.62, 71.10, -20.15, -8.07, -46.58 | 33.6° |
| grasp | 34.69, -75.75, 102.85, 78.93, -16.68, 9.45, -49.24 | 26.3° |
| grasp_red | -2.21, -82.27, 96.55, 61.83, -9.91, 2.06, -51.41 | 19.7° |
| grasp_blue | 61.15, -86.45, 88.86, 61.00, -2.56, -2.71, 20.06 | 15.5° |
| lift | 30.93, -76.30, 69.72, 67.44, -5.62, -23.70, -48.15 | 20.3° |
| transit | 71.58, -61.87, 76.62, 76.22, -25.51, -18.09, -20.14 | 25.9° |
| place_hover | 90.77, -73.38, 80.92, 60.88, -10.56, -15.75, 19.10 | 28.3° |
| place | 93.70, -85.42, 87.76, 70.09, -3.56, -3.66, 15.41 | 16.6° |

算法是"多起点随机射击 + 软限位阻尼最小二乘"：随机撒 500 组初始关节角，
每组迭代 320 次，取残差最小的解，再在候选里按余量 / 关节位移量打分。

## 抓取-放置任务（`pick_place.py`）

把指定颜色的方块夹起来搬进篮子 `(-0.04, 0.42)`。11 段动作：

| # | 关键帧 | 夹爪 | 动作 |
| --- | --- | --- | --- |
| 1 | hover | 张开 | 移到目标方块正上方 |
| 2 | grasp | 张开 | 下降到方块中心 |
| 3 | grasp | **闭合** | 两指夹紧（4 个 pad↔cube 接触） |
| 4 | lift | 闭合 | 竖直提起 150 mm |
| 5 | transit | 闭合 | 平移到篮子上方 |
| 6 | place_hover | 闭合 | 到篮子正上方 120 mm |
| 7 | place | 闭合 | 下降把方块放进篮子 |
| 8 | place | **张开** | 松手释放 |
| 9 | place_hover | 张开 | 竖直退出篮子（避免刮到侧壁） |
| 10 | transit | 张开 | 原路返回 |
| 11 | ready | 张开 | 回到待命姿态 |

实测结果（`./run.sh place`）：

```
绿方块落点 [-0.038 0.419 0.788]
篮子中心 (-0.040, 0.420)   底板顶面 z=0.748   内半宽 0.070
偏离篮心 2.1 mm（限 30.0）
底面高差 0.0 mm（限 12.0）
结果：PASS
```

### 抓不同颜色：`--cube red|green|blue`

三块方块各有一条独立的关键帧链，键名带颜色前缀（`red_hover`、`blue_grasp` …），
所以同一个场景里三套动作互不干扰：

```bash
./run.sh pick --cube red        # 交互窗口演示搬红块
./run.sh pick --cube blue       # 交互窗口演示搬蓝块
python3 pick_place.py --cube red --check      # 无窗口自检
python3 pick_place.py --all --check           # 依次搬红、绿、蓝，全部判定
python3 pick_place.py --all --shots           # 出 21 张三色过程图
```

实测三色全部 PASS：

| 方块 | 偏航 | 抓取余量 | 落点 | 偏离篮心 |
| --- | --- | --- | --- | --- |
| 红 `cube_red` | +12° | 9.9° | `[-0.038 0.436 0.788]` | 16.1 mm |
| 绿 `cube_green` | −8° | 19.4° | `[-0.039 0.418 0.788]` | 2.1 mm |
| 蓝 `cube_blue` | +20° | 17.8° | `[-0.039 0.419 0.788]` | 1.3 mm |

`poses.json` 共 19 个位姿 = `ready` + 3 色 × 6 个关键帧，全部 `pos_err=0.00 mm`、
`tilt=0.00°`、夹持面偏航误差 `0.00°`。

#### 踩过的坑一：IK 分支必须锚定 ready

给三色解 IK 时如果每个都从 `q_ref=None` 随机起解，红/蓝会解到 joint5 为正的
**另一个分支**（手臂绕到身后），关节空间要扫 146°，动力学跟不过去：
`blue` 从 `blue_hover`（joint5=+141°）到 `blue_grasp`（joint5=−5°），实测指尖偏 72 mm。

修法：`solve_task(cube, q_ref=q_ready)`，且链内递进锚定（每解完一个位姿就用它
当下一段的 `q_ref`）；`solve()` 的参考姿态惩罚权重从 0.25 提到 **2.0**。
修好后三色链的 joint5 都在 −27°…+5° 区间，不再翻分支。

#### 踩过的坑二：夹爪偏航方向反了（最关键）

`yaw_rad()` 用 `atan2(padl.x−padr.x, padl.y−padr.y)` 量的是"从 +y 转向 +x"的
**顺时针**角；而 `cube_pose()` 的 `yaw_c` 用 `atan2(R[1,0], R[0,0])` 是**逆时针**角。
早期直接传 `yaw_tgt=yaw_c`，夹爪就朝反方向转，两指连线与方块面法向的夹角变成
**2×偏航角**：

| 方块 | 偏航 | 夹角 | 沿夹持方向投影宽 | 夹爪开度 |
| --- | --- | --- | --- | --- |
| 红 | +12° | 24° | 105.6 mm | 100 mm |
| 绿 | −8° | 16° | 99.0 mm | 100 mm |
| 蓝 | +20° | 40° | **112.7 mm** | 100 mm |

蓝块投影宽 112.7 mm > 100 mm 开度，下降时指尖必然顶到方块上棱角，
joint2 力矩饱和到 −24 Nm，指尖卡在 z=0.834（方块顶面 0.819）再也下不去。
红块 105.6 mm 也超了，只是偏得少、勉强擦过去。

修法：`yaw_tgt = -yaw_c`。修好后三色"两指连线 vs 方块局部 y 轴"夹角全部
**0.0°**（`tools/diag_align.py` 可直接复验）。

#### 顺带纠正：`grasp` 阶段的夹爪开度不是越小越好

把 `GRIP_OPEN` 从 0.050 提到 0.055（110 mm）对红/绿毫无影响，对蓝块也只是
从"卡死"变成"擦着下去"——真正的问题是偏航没对齐，不是开度不够。

### 平滑轨迹：为什么不能用阶跃 ctrl

第一版直接 `d.ctrl = q_target`，关节在第一个仿真步就要求满速，
末端加速度达 15 m/s² 量级，方块会在指尖之间**蹭着滑走**：

| 阶段 | 阶跃 ctrl 的滑移 | 余弦平滑后的滑移 |
| --- | --- | --- |
| lift | 3.9 mm | 0.3 mm |
| transit | 累计 13.5 mm | 累计 0.6 mm |
| place_hover | 累计 **22.2 mm** | 累计 1.0 mm |
| 最终偏离篮心 | **24.8 mm** | **2.1 mm** |

`move()` 用 `s(t) = 0.5·(1 − cos(π·t/T))` 插值，起停加速度都为 0，
再在末态保持 0.4 s 把残余跟踪误差收敛掉。滑移全部发生在每段运动的
**起始 0.2–0.4 s**，用逐帧诊断（`tools/diag_slip2.py`）定位出来的。

### 伺服与重力

- 位置执行器 `kp/kv` 由根部到末端递减：J1/J2 `400/12`、J3 `300/9`、J4 `200/6`、
  J5 `120/4`、J6 `50/2`、J7 `25/1`；夹爪 `3000/6`。
- 每个 body 加 `gravcomp="1"`（等价真实控制器的重力补偿），
  ready 姿态保持 3 s 的最大漂移从 1.43° 降到 **0.32°**。
- 抬起动作是"保持夹紧 + 竖直伺服回 hover"，**不是**转 J2——
  实测转 J2 会让腕部下移 1.1 mm，抓取失败。

## 验证

`./run.sh validate` 跑 6 段检查，当前全部 PASS：

| 检查 | 结果 |
| --- | --- |
| 结构 | 7 关节 + 7 执行器 + 夹爪关节/执行器全部绑定；本体 4.800 kg + 夹爪 0.500 kg |
| 限位 | 7 个关节与手册逐项一致 |
| 运动学 | 9 个关键帧指尖中点精确命中目标，工具 z 轴全为 (0,0,-1) |
| 稳定性 | ready 保持 3 s，最大漂移 0.318° |
| 抓取 | hover → grasp（偏差 3.5 mm）→ 4 个 pad↔cube 接触 → 抬起 **Δz = +0.117 m** |
| 渲染 | 机械臂在 4 个机位的画面占比 1.54% / 4.66% / 4.48% / 2.07% |

抓取-放置另有独立自检 `./run.sh place`（方块落进篮子返回 0），
以及 21 张过程图 `./run.sh pp-shots all`（三种颜色各 7 张，带颜色前缀避免覆盖）：

| 图 | 内容 |
| --- | --- |
| `pp_<col>_1_grasped.png` | 夹爪闭合并夹住方块 |
| `pp_<col>_2_lifted.png` | 竖直提起 150 mm |
| `pp_<col>_3_transit.png` | 横移到篮子上方途中 |
| `pp_<col>_4_above_basket.png` | 悬停在篮子正上方 120 mm |
| `pp_<col>_5_placed.png` | 放进篮子（俯视，越过 90 mm 侧壁） |
| `pp_<col>_6_done.png` | 机械臂回到 ready，方块留在篮内 |
| `pp_<col>_7_basket_top.png` | 篮子俯视特写，方块躺在篮底正中 |

其中 `<col>` ∈ `red` / `green` / `blue`。只出单色用 `./run.sh pp-shots red`。

判定不靠肉眼：像素占比用"关掉 geomgroup[2] 前后两次渲染差分"客观测量；
过程图里目标方块的颜色像素也用 `tools/check_color_px.py` 逐张统计过
（红 477–24785 px、绿 1553–27291 px、蓝 1835–24216 px，全部 >0）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `build_nero.py` | **生成器**：读 URDF + CSV + poses.json，写出 `nero_arm.xml` 与 `nero_table_scene.xml` |
| `nero_table_scene.xml` | 生成物：桌子 + 三色方块 + 收纳篮 + NERO 机械臂（主场景） |
| `nero_arm.xml` | 生成物：只有机械臂，基座在原点（便于单独复用） |
| `solve_poses.py` | 6 残差阻尼最小二乘 IK，写 `poses.json`（19 个关键帧 = ready + 3 色 × 6） |
| `poses.json` | 19 个关键帧的关节角 |
| `pick_place.py` | **抓取-放置任务**：11 段平滑轨迹，`--cube red\|green\|blue` / `--all` / `--check` / `--viewer` / `--shots` |
| `validate_nero.py` | 6 段全量校验 |
| `render_showcase.py` | 成果图（自动取景） |
| `render_detail.py` | 夹爪抓取特写 |
| `view_scene.py` | 交互窗口 / `--demo` 演示 / `--check` 自检 |
| `view.py` | 第一版：只有桌子 + 三色方块 |
| `check_colors.py` | 校验渲染图里确实有三种不同颜色 |
| `run.sh` | 一键入口（`pick`/`place`/`pp-shots` 都支持 `red\|green\|blue\|all`） |
| `assets/nero/` | STL 网格 + 原始 CSV |
| `out/` | 渲染结果（含 `pp_<col>_*.png` 三色过程图） |
| `tools/` | 诊断脚本（FK 交叉验证、抓取/滑移/对齐诊断、篮子筛选等） |

## 复现

```bash
cd ~/NERO
/home/zhangb/so101_mujoco/.venv/bin/python solve_poses.py   # 重新解位姿 → poses.json
/home/zhangb/so101_mujoco/.venv/bin/python build_nero.py    # 重新生成 XML
/home/zhangb/so101_mujoco/.venv/bin/python validate_nero.py # 校验
/home/zhangb/so101_mujoco/.venv/bin/python pick_place.py --all --check   # 三色抓放自检
```

**顺序很重要**：`solve_poses.py` 从现有 XML 读方块/篮子的真实位姿，`build_nero.py`
再把 `poses.json` 写进 XML。所以改了桌子/篮子/方块的几何后必须
**build → solve → build**，否则新关键帧不会进 XML（运行时报
`KeyError: Invalid name 'transit'`）。

## 已知限制

- 碰撞体是简化几何（capsule/box），不是精细网格，不能用来做高精度碰撞检测。
- 夹爪没有做真的"力控抓取"，靠位置伺服 + 指尖摩擦；方块 0.30 kg，抓 3 kg 负载需另做。
- 桌面没有建摩擦参数调优，方块静置稳定但快速拖动会滑。
- 篮子是 5 个 box 拼的，没有做圆角/镂空；内腔 140 mm 装一块 80 mm 方块够用，装不下两个。
- 未接入 `pyAgxArm` 真实控制器；执行器是 MuJoCo `position` 类型，
  真实机器人是 MIT 模式（`t_ff` 全关节 ±16 N·m）。

## 排错：篮子相关

**现象**：`place` / `place_hover` / `transit` 三个位姿全部「无解」，`basket_frame()` 报
`floor_top = 0.008`、放置点 z ≈ 0.05（比桌面还低）。

**原因**：`basket` 这个 body 的 `pos` 写成了 `"x y 0"`，漏了 `TABLE_Z`，
整个篮子落到世界原点的地面上。`floor_top` 是从 `d.geom_xpos` 读的，
所以读出来是真实的世界坐标 z=0.008。

**修法**：`basket_xml()` 里 `pos="%.3f %.3f %.3f" % (BASKET_XY[0], BASKET_XY[1], TABLE_Z)`。
改完必须 build → solve → build（见「复现」）。

**现象**：明明目测方块在篮子里，`pp_6_done.png` 里却看不见。

**原因**：篮子侧壁 90 mm 高、方块 80 mm 高，从 el=-16° 的平视机位看过去
方块被侧壁完全挡住（绿像素只剩 4 个）。

**修法**：拍放置结果要用俯视，`el = -52°`（`pp_5_placed`）或 `-68°`（`pp_7_basket_top`）。

## 排错：交互窗口里"只显示一部分机械臂"

**现象**：`./run.sh` 打开窗口后只看到底座或一小段连杆。

**原因**：MuJoCo `launch_passive` 的 free 相机初值是硬编码的
`lookat=(0,0,0)`、`distance=2.0`、`azimuth=90`、`elevation=-45`，
视线指向世界原点——也就是地面。而机械臂装在桌面 z=0.74 上，
整臂顶端在 z≈1.25，实测 **96.7% 的网格顶点被顶出画面上边缘**。

注意：在 XML 里写 `<statistic center=... extent=.../>` **不能**修正这个行为
（实测 `m.stat.center` 变了，但 viewer 相机仍从 `(0,0,0)` 起步）。

**修法**：`view_scene.py` 的 `frame_camera()` 在 `launch_passive` 之后
按机械臂实际包围盒重设 `v.cam.lookat / distance / azimuth / elevation`。
实测采样点入画比例 **3.3% → 100%**。

```python
lo, hi = arm_bounds(m, d)                       # geomgroup 2 的网格顶点
c = (lo + hi) / 2.0
span = np.linalg.norm(hi - lo)
v.cam.lookat[:] = c
v.cam.distance = 1.15 * span / (2 * np.tan(np.radians(fovy) / 2))
v.cam.azimuth, v.cam.elevation = 55.0, -18.0
```

在窗口里随时可以用鼠标自由旋转；`Tab` 切场景预设相机（`overview` /
`arm_front` / `arm_side` / `top`），这四个机位的取景本来就是对的。
