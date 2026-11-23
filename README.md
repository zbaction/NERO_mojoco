# NERO 机械臂 MuJoCo 场景

AgileX **NERO** 七自由度机械臂 + 两指夹爪，装在木桌上，桌面上摆着红/绿/蓝三个方块，
旁边放着一个木色收纳篮，桌面外侧立着一台 **RGB-D 深度相机**。
模型由官方 URDF/质量表逐项转换为 MuJoCo MJCF，
机械臂可以真的走过去**夹起方块、搬过去、放进篮子里**；
深度相机可以**从点云里测出方块的中心位置**（误差 1.3–1.9 mm）。

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
./run.sh pick all     # 依次搬红 → 绿 → 蓝，三块最终同时留在篮内
./run.sh check        # 无窗口自检，抓取成功返回 0
./run.sh place        # 无窗口自检：绿方块落进篮子返回 0
./run.sh place blue   # 同上，但搬蓝方块（也支持 red / all）
./run.sh place all    # 依次搬三块，并断言先放的不会被后放的挪走
./run.sh validate     # 7 段全量校验（含深度相机）
./run.sh depth        # 深度相机拼版图：RGB / 深度伪彩 / 俯视点云 / 测量数值
./run.sh depth-check  # 深度相机测量自检，三块方块中心误差 < 6 mm 返回 0
./run.sh shots        # 重出成果图到 out/
./run.sh pp-shots all # 重出三色各 7 张过程图到 out/
./run.sh detail       # 夹爪特写图到 out/
./run.sh desk         # 只看第一版的桌子 + 三色方块
```

> `run.sh` 里 `pick` / `place` / `pp-shots` 的第一个参数如果是
> `red` / `green` / `blue` / `all`，会自动翻译成 `--cube <颜色>` 或 `--all`；
> 其他参数原样透传（例如 `./run.sh pick --cube blue`）。

交互窗口里鼠标左键旋转、右键平移、滚轮缩放，`Tab` 切换预设相机，
空格暂停，`Esc` 退出。

## 场景内容

- **桌子**：桌面 0.8 m × 1.2 m，上表面 z = 0.740 m，四条腿 + 横撑。
- **机械臂**：AgileX NERO，7 个转动关节 + 末端两指夹爪，基座装在桌面
  `(-0.30, 0.0, 0.74)`，用一块 0.10 m × 0.10 m × 0.014 m 的转接板固定。
- **方块**：边长 **0.080 m**（8 cm），质量各 0.30 kg，带 `freejoint` 参与真实物理，
  初始绕 z 轴各有一点偏转：红 +12°、绿 -8°、蓝 +20°，位置分别为
  `(0.16, -0.26)`、`(0.18, 0)`、`(0.15, +0.26)`。
- **收纳篮**：中心 `(-0.16, 0.42)`，外廓 0.296 m 见方、侧壁高 0.090 m、壁厚 8 mm，
  内腔 **280 mm × 280 mm**，底板顶面 z = 0.748 m。5 个 box 拼成（底板 + 四面侧壁），
  也是真实碰撞体，方块扔进去会真的撞在壁上、落在底板上。
  篮内用三个 `<site>` 标出红/绿/蓝各自的放置格位（三角形排布，两两间距 120 mm），
  这样三块 80 mm 方块可以**同时**留在篮内而互不干扰。
- **深度相机**：一台 RGB-D 相机架在三脚架上，站在桌面外侧
  `(0.57, 0.57)` 的地面上，光心高 1.20 m、俯角约 31°、垂直视场 58°，
  分辨率 848×480。它是一台**真实的传感器模型**——不是"从场景里读真值"，
  而是走 MuJoCo 深度渲染 → 反投影 → 点云 → 测出方块中心，
  三块方块的实测中心误差 1.3–1.9 mm。详见[深度相机](#深度相机rgb-d)一节。

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

把指定颜色的方块夹起来搬进篮子里**自己的格位**。逻辑上是 11 个关键帧，
执行时由 `plan_for()` 合并成 **3 条连续轨迹 + 2 次原地夹爪动作**
（轨迹连续、只在夹爪开合处停，详见「平滑轨迹」一节）：

| # | 关键帧 | 夹爪 | 动作 |
| --- | --- | --- | --- |
| 1 | hover | 张开 | 移到目标方块正上方 |
| 2 | grasp | 张开 | 下降到方块中心 |
| 3 | grasp | **闭合** | 两指夹紧（8 个 pad↔cube 接触） |
| 4 | lift | 闭合 | 竖直提起 150 mm |
| 5 | transit | 闭合 | 平移到篮子上方 |
| 6 | place_hover | 闭合 | 到篮子正上方 120 mm |
| 7 | place | 闭合 | 下降把方块放进篮子 |
| 8 | place | **张开** | 松手释放 |
| 9 | place_hover | 张开 | 竖直退出篮子（避免刮到侧壁） |
| 10 | transit | 张开 | 原路返回 |
| 11 | ready | 张开 | 回到待命姿态 |

合并后的执行顺序（`plan_for()` 的 5 项）：

| 步 | 类型 | 途经关键帧 | 夹爪 |
| --- | --- | --- | --- |
| 1 | path | 当前姿态 → hover → grasp | 张开 |
| 2 | dwell | — | 闭合 |
| 3 | path | grasp → lift → transit → place_hover → place | 闭合 |
| 4 | dwell | — | 张开 |
| 5 | path | place → place_hover → transit → ready | 张开 |

其中 `hover` 和 `place_hover` 是**刹车点**（`hold`）：轨迹在这里必须
零速零加速度通过，否则会带着速度撞上方块或篮底。

实测结果（`./run.sh place`）：

```
绿方块落点 [-0.099 0.338 0.788]
偏离格位 (-0.100, 0.340) 2.3 mm（限 20.0）
底面高差 0.0 mm（限 12.0）
结果：PASS
```

连着搬三块（`./run.sh pick all`）：

```
red   红方块 落点 [-0.219 0.337 0.788]  偏离格位 2.9 mm  -> PASS
green 绿方块 落点 [-0.099 0.338 0.788]  偏离格位 2.3 mm  -> PASS
blue  蓝方块 落点 [-0.159 0.500 0.788]  偏离格位 1.2 mm  -> PASS

搬运过程中已放置方块的最大水平扰动 0.00 mm（限 5.0）
同时检查三块是否都在篮内：
  red   [-0.219 0.337 0.788]  偏离格位 2.9 mm  -> PASS
  green [-0.099 0.338 0.788]  偏离格位 2.3 mm  -> PASS
  blue  [-0.159 0.500 0.788]  偏离格位 1.2 mm  -> PASS
总结果：PASS
```

整个流程仿真 **13.9 s**（连续轨迹，中途只在 2 次夹爪开合处停）。
`./run.sh pick` 在交互窗口里按 **1.00× 实时**播放。

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

| 方块 | 偏航 | 抓取余量 | 落点 | 偏离自己的格位 |
| --- | --- | --- | --- | --- |
| 红 `cube_red` | +12° | 9.9° | `[-0.218 0.339 0.788]` | 2.3 mm |
| 绿 `cube_green` | −8° | 19.4° | `[-0.099 0.338 0.788]` | 2.3 mm |
| 蓝 `cube_blue` | +20° | 17.8° | `[-0.159 0.499 0.788]` | 2.0 mm |

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

### 平滑轨迹：从"一段一顿"到连续 C2 轨迹

#### 第一版：阶跃 ctrl（最糟）

最初直接 `d.ctrl = q_target`，关节在第一个仿真步就要求满速，末端加速度
达 15 m/s² 量级，方块会在指尖之间**蹭着滑走**：

| 阶段 | 阶跃 ctrl 的滑移 | 余弦平滑后的滑移 |
| --- | --- | --- |
| lift | 3.9 mm | 0.3 mm |
| transit | 累计 13.5 mm | 累计 0.6 mm |
| place_hover | 累计 **22.2 mm** | 累计 1.0 mm |
| 最终偏离放置点 | **24.8 mm** | **2.1 mm** |

`move()` 用 `s(t) = 0.5·(1 − cos(π·t/T))` 插值，起停加速度都为 0，
再在末态保持 0.4 s 把残余跟踪误差收敛掉。滑移全部发生在每段运动的
**起始 0.2–0.4 s**，用逐帧诊断（`tools/diag_slip2.py`）定位出来的。

#### 第二版：11 段各自平滑，但段间必停（看着一顿一顿）

`phases_for()` 把任务切成 11 段，每段都是"启动→到点→保持 0.4 s"。
曲线本身很光滑，但**每段都要停死再重启**。`tools/diag_smooth.py` 实测：

```
仿真时长 27.90 s   墙钟 0.34 s   倍速 82.64x
  最大关节速度 0.69 rad/s
  最大关节加速度 90.46 rad/s²
  最大 ctrl 速度 31.93 rad/s     ← 段边界硬跳
  停顿次数 15 次，累计 8.11 s     ← 占总时长 29%
ctrl 速度突变：t=18.296 s / 20.196 s 处各 31.9 rad/s
```

两个独立问题：

1. **每段都要停车** → 15 次停顿、累计静止 8.11 s，肉眼看就是"动一下停一下"。
2. **`--viewer` 是 82 倍速快放** → `_step_loop` 原来"能跑多快跑多快"，
   仿真 27.9 s 只花 0.34 s 墙钟，画面里机械臂一闪而过。

顺带确认**伺服本身没问题**（`tools/diag_track.py`）：transit 段指令位移
52.3°，跟踪误差峰值 1.02°、RMS 0.44°。所以不该动 kp/kv。

#### 第三版（当前）：`plan_for()` + 五次 Hermite 连续轨迹

改成 **3 条连续轨迹 + 2 次原地夹爪动作**，只在真正要动夹爪的地方停：

| 步 | 动作 | 途经点 | 夹爪 |
| --- | --- | --- | --- |
| 1 | 接近并下降 | `当前姿态 → <col>_hover → <col>_grasp` | 张开 |
| 2 | 闭合夹爪 | — | 闭合 |
| 3 | 提起 → 平移 → 放进篮子 | `<col>_grasp → lift → transit → place_hover → place` | 闭合 |
| 4 | 张开夹爪释放 | — | 张开 |
| 5 | 退出篮子 → 回待命 | `<col>_place → place_hover → transit → ready` | 张开 |

关键实现（`pick_place.py`）：

- **`_hermite5()`**：五次 Hermite 同时匹配两端的位置、速度、加速度，
  整条曲线 **C2 连续**。三次 Hermite 只做到 C1，节点处加速度会跳变
  （jerk 无穷大），高速时看得见"到点抽一下"。
- **`_sample_path()`**：途经点速度取 Catmull-Rom 时域版本
  `M[i] = damp·(Q[i+1]−Q[i−1])/(knots[i+1]−knots[i−1])`（端点速度为 0），
  节点加速度取相邻两段弦加速度之差，两者都参与插值。
- **`_build_path()`**：两级重试 —— ① 关节限位越界则衰减 `damp`
  （1.0→0.6→0.35→0.2→0.1→0）；② 速度超 `JV_PEAK` 则整条按比例放慢。
- **`hold`（刹车点）**：`hover` / `place_hover` 必须**零速零加速度**通过。
  这是必须的 —— 连续轨迹如果带着速度冲过 hover 直接扎进方块，指尖会从
  侧面撞上方棱，蓝块（+20° 偏航、投影宽 112.7 mm）当场被撞倒
  （实测 tilt 40°→90°，`Rzz` 从 0.765 掉到 0.000）。
- **`slow`（放慢段）**：下降夹取、下放落位这两段时长 ×2.2，落点更稳。
- **段边界 ctrl 平滑接管**：轨迹起点从"当前 ctrl"用 0.15 s 余弦混到
  轨迹首点，`dwell()` 也从"当前 ctrl"混到实测 qpos。原来直接把 `qpos`
  塞进 `ctrl` 会留 0.44° 台阶。
- **`_step_loop()` 实时节拍**：viewer 模式下按 60 Hz 墙钟 `time.sleep`
  同步，仿真时间 ≈ 墙钟时间。实测墙钟 13.88 s / 仿真 13.88 s（**1.00×**，
  原来是 82× 快放）。

#### 效果（`tools/plot_smooth.py`，出图 `out/smooth_compare.png`）

| 指标 | 旧版（11 段） | 新版（连续 C2） |
| --- | --- | --- |
| 总时长 | 27.90 s | **13.87 s** |
| 指尖速度峰值 | 0.19 m/s | 0.43 m/s |
| 关节速度峰值 | 32.94 °/s | 48.24 °/s（限速 180–225 °/s） |
| 关节加速度峰值 | 956.6 °/s² | 1942.1 °/s² |
| **ctrl 速度峰值** | **111.77 °/s** | **48.35 °/s** |
| 指尖最大加速度 | 15.9 m/s² | 23.1 m/s² |
| 中途停顿 | 15 次 / 累计 8.11 s | 8 次 / 累计 2.52 s |
| `--viewer` 播放倍速 | 82.6×（快放） | **1.00×（实时）** |

新版剩下的 8 次"低速段"全部是**设计上该停的地方**（夹爪开合 2 次、
hover 刹车 2 次、place_hover 刹车 2 次、末态收敛），不是轨迹抖动。

`tools/diag_track2.py` 跟踪误差：整体峰值 2.34°、RMS 0.57°。

### 伺服与重力

- 位置执行器 `kp/kv` 由根部到末端递减：J1/J2 `400/12`、J3 `300/9`、J4 `200/6`、
  J5 `120/4`、J6 `50/2`、J7 `25/1`；夹爪 `3000/6`。
- 每个 body 加 `gravcomp="1"`（等价真实控制器的重力补偿），
  ready 姿态保持 3 s 的最大漂移从 1.43° 降到 **0.32°**。
- 抬起动作是"保持夹紧 + 竖直伺服回 hover"，**不是**转 J2——
  实测转 J2 会让腕部下移 1.1 mm，抓取失败。

## 深度相机（RGB-D）

场景里装了一台深度相机（`depth_cam`），用来做**感知**而不是读真值：
它走的是和真实 RGB-D 完全一样的链路 —— 渲染深度图 → 用内参反投影成点云 →
在点云上分割颜色 → 拟合方块中心。抓取位姿仍然来自 IK 离线解，
但测量精度已经过量化验证，可以拿它做后续的视觉伺服。

### 安装

| 项 | 值 |
| --- | --- |
| 相机名 | `depth_cam`（世界坐标固定相机，**不跟夹爪动**） |
| 光心 | `(0.57, 0.57, 1.20)` m（桌面外侧地面上的三脚架） |
| 看向 | `(0.00, 0.20, 0.80)`，俯角 30.7° |
| 分辨率 / 视场 | 848 × 480，纵向 fovy 58°（`fx = fy = 432.97`，`cx = 424`、`cy = 240`） |
| 深度范围 | 0.05–2.5 m 有效（`map.zfar = 30`，天空盒像素深度是 `stat.extent×zfar = 63`，按上限剔除） |

相机位姿是**网格搜索**出来的（`tools/probe_cam2.py`~`probe_cam11.py`）：
在"站在桌面外侧"的前提下，要求 ready 姿态下三块方块顶面 + 篮子四角共 7 个工作点
全部入画且不被机械臂遮挡、19 个关键帧总遮挡最少、方块 ≥32 px、俯角 28–52°。
247 个可行解里最优的那个总遮挡只有 3 次（都是夹爪自己压住目标方块，属正常遮挡），
方块 44 px、边距 176 px。

几何上它是一个真的能看见的物体：三脚架（底座 + 立柱 + 3 根斜撑）+ 云台壳体 + 镜头，
`dcam_lens` 带 `contype=0`（纯视觉），`<site name="dcam_optical">` 标出光心。
整机外观见 `out/depth_cam_scene.png`（`python3 render_depthcam_shot.py` 生成，
自动取景把桌子、机械臂、三脚架三者一起框住）。

### 用法

```bash
./run.sh depth          # out/depth_camera.png：2×2 拼版
./run.sh depth --multi  # 额外出 ready / green_hover / green_grasp 三个位姿的深度图
./run.sh depth-check    # 测量自检，三块方块中心误差 < 6 mm 返回 0
./run.sh validate       # 全量校验里的第 7 段就是深度相机
```

`out/depth_camera.png` 是四宫格：

| 区块 | 内容 |
| --- | --- |
| 左上 | RGB 视图，叠测量中心十字 + 真实位姿方框（重合即正确） |
| 左下 | 深度伪彩（0.3–2.4 m，turbo 配色） |
| 右上 | 俯视散点图：真实点云打底 + 测量中心●/真实中心○ |
| 右下 | 数值面板：像素数、点数、测得/真实中心、误差 |

### 代码

| 文件 | 说明 |
| --- | --- |
| `depth_cam.py` | `DepthCamera`：`capture()` / `pointcloud()` / `unproject()` / `project()` / `intrinsics()` / `pose()` |
| `depth_view.py` | 测量与自检（`--check` / `--all-poses`），出 `out/depth_rgb.png`、`depth_colormap.png`、`depth_pointcloud.ply` |
| `depth_show.py` | 出拼版图 `out/depth_camera.png`（`--multi` 加多姿态图） |

### 实测精度（`./run.sh depth-check`）

```
red    2912 px  2629 点  [0.161 -0.259 0.822]  vs 真实 [0.160 -0.260 0.780]  Δxy 1.3 mm  Δz 1.6 mm
green  3690 px  3377 点  [0.182  0.000 0.821]  vs 真实 [0.180  0.000 0.780]  Δxy 1.5 mm  Δz 1.3 mm
blue   4422 px  4134 点  [0.152  0.261 0.821]  vs 真实 [0.150  0.260 0.780]  Δxy 1.9 mm  Δz 1.2 mm
桌面 132959 点，z 中位数 0.7405 m（真值 0.740），误差 0.5 mm
```

`Δz` 是"顶面中心 vs 方块几何中心"的差，等于半个边长 40 mm 减去测量点偏移，
统计上稳定在 1.2–1.6 mm；`Δxy` 才是真正的位置误差，**1.3–1.9 mm**。

### 三个关键实现点

**① 深度图是"沿光轴的 z"，不是欧氏距离**
`Renderer.enable_depth_rendering()` 返回 z；反投影公式是
`xc=(u-cx)/fx*z`、`yc=-(v-cy)/fy*z`、`zc=-z`（相机系 +z 朝后，所以取负），
再 `pw = R @ [xc,yc,zc] + p0`，其中 `R = d.cam_xmat[cid].reshape(3,3)`（列 = 相机 x/y/z 轴）。

**② 背景不是 `nan` 或 `inf`**
天空盒像素深度恒为 `stat.extent * map.zfar`（本场景 = 63.0），所有像素都是有限值。
所以剔除背景必须用"深度 > max_depth"，`np.isfinite` 一点用没有。
改 `stat.extent`、`map.zfar`、开关 skybox **都不影响前景像素**，只影响这个背景常数。

**③ 测中心必须用分位中点，不能用均值**
立方体朝相机的那两面侧壁会有一条窄带被颜色掩膜选进"顶面点集"，
用 `mean()` 会带来 **+7 mm 的系统性偏移**（三块方块全往 +x+y 偏）。
改用 `lo/hi = np.percentile(top[:,:2], [1,99], axis=0)`、`center = 0.5*(lo+hi)`
（即包围盒中心）后，误差降到 1.3–1.9 mm。选点高度也从 `z_top-0.008` 收紧到 `z_top-0.003`。

### 踩过的坑

- **相机 `<camera>` 必须挂在 `<worldbody>` 下**。一度挂进 `gripper_base` body 里，
  结果相机跟着夹爪动，渲染出来只剩一片渐变天空（RGB mean 63–67、std 9–10）。
- **`mujoco.mj_ray()` 是 9 个参数**，`geomid` 必须传 `np.zeros(1, np.int32)`；
  只传 7 个会 `TypeError: incompatible function arguments`。
- **`mjtVisFlag` 里没有 `mjVIS_SKYBOX`**，`MjvOption.flags` 长度 31；
  `mujoco` 也没有 `mj_project`/`mj_unproject` 这类 API。
- **PIL `ImageDraw.rectangle` 要求 `[x0,y0,x1,y1]` 且 `y1 >= y0`**，
  直接传两个对角点会 `ValueError: y1 must be greater than or equal to y0`。

## 验证

`./run.sh validate` 跑 7 段检查，当前全部 PASS：

| 检查 | 结果 |
| --- | --- |
| 结构 | 7 关节 + 7 执行器 + 夹爪关节/执行器全部绑定；本体 4.800 kg + 夹爪 0.500 kg |
| 限位 | 7 个关节与手册逐项一致 |
| 运动学 | 19 个关键帧（ready + 3 色 × 6）指尖中点和夹持面朝向都精确命中 |
| 稳定性 | ready 保持 3 s，最大漂移 0.318° |
| 抓取 | hover → grasp（偏差 3.4 mm）→ 8 个 pad↔cube 接触 → 抬起 **Δz = +0.117 m** |
| 渲染 | 机械臂在 4 个机位的画面占比 1.59% / 4.80% / 4.52% / 2.05% |
| 深度相机 | 848×480 fovy 58°，有效几何 64.3%，点云桌面高度误差 **0.5 mm**，三块方块中心误差 1.3–1.9 mm |

运动质量另有三个量化检查：

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 实时播放 | `python3 tools/test_realtime.py` | 墙钟 13.88 s = 仿真 13.88 s，**1.00× 实时**（改前 82.6× 快放） |
| 轨迹平滑 | `python3 tools/diag_smooth.py green` | `ctrl` 最大单步跳变 0.44°、>0.05 rad 跳变 **0 处**；停顿 8 次 / 2.52 s（改前 15 次 / 8.11 s） |
| 跟踪误差 | `python3 tools/diag_track2.py green` | 指令 vs 实际峰值 2.34°、RMS 0.57° |

新旧对比曲线 `python3 tools/plot_smooth.py` → `out/smooth_compare.png`。

抓取-放置另有独立自检 `./run.sh place`（方块落进篮子返回 0）、
`./run.sh place all`（依次搬红→绿→蓝，并校验前一块不被后续搬运挪动），
以及 21 张过程图 `./run.sh pp-shots all`（三种颜色各 7 张，带颜色前缀避免覆盖）：

| 图 | 内容 |
| --- | --- |
| `pp_<col>_1_grasped.png` | 夹爪闭合并夹住方块 |
| `pp_<col>_2_lifted.png` | 竖直提起 150 mm |
| `pp_<col>_3_transit.png` | 横移到篮子上方途中 |
| `pp_<col>_4_above_basket.png` | 悬停在篮子正上方 120 mm |
| `pp_<col>_5_placed.png` | 放进篮子（俯视，越过 90 mm 侧壁） |
| `pp_<col>_6_done.png` | 机械臂回到 ready，方块留在篮内 |
| `pp_<col>_7_basket_top.png` | 篮子俯视特写，方块躺在自己的格位上 |

其中 `<col>` ∈ `red` / `green` / `blue`。只出单色用 `./run.sh pp-shots red`。

搬完三块后的合影：`out/basket_three_cubes.png`（俯视，三块方块各在自己的格位上）。
重新生成用 `python3 tools/shot_basket_all.py`。

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
| `pick_place.py` | **抓取-放置任务**：3 条连续 C2 轨迹 + 2 次夹爪动作，实时播放；`--cube red\|green\|blue` / `--all` / `--check` / `--viewer` / `--shots` |
| `validate_nero.py` | 7 段全量校验（结构/限位/运动学/稳定性/抓取/渲染/深度相机） |
| `depth_cam.py` | 深度相机封装：深度渲染、反投影、点云、投影 |
| `depth_view.py` | 深度测量与自检（`--check` 出 PLY 点云） |
| `depth_show.py` | 深度相机拼版成果图 |
| `render_showcase.py` | 成果图（自动取景） |
| `render_detail.py` | 夹爪抓取特写 |
| `view_scene.py` | 交互窗口 / `--demo` 演示 / `--check` 自检 |
| `view.py` | 第一版：只有桌子 + 三色方块 |
| `check_colors.py` | 校验渲染图里确实有三种不同颜色 |
| `run.sh` | 一键入口（`pick`/`place`/`pp-shots` 都支持 `red\|green\|blue\|all`；`depth` 出深度相机图） |
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
/home/zhangb/so101_mujoco/.venv/bin/python depth_view.py --check        # 深度相机测量自检
```

**顺序很重要**：`solve_poses.py` 从现有 XML 读方块/篮子的真实位姿，`build_nero.py`
再把 `poses.json` 写进 XML。所以改了桌子/篮子/方块的几何后必须
**build → solve → build**，否则新关键帧不会进 XML（运行时报
`KeyError: Invalid name 'transit'`）。

## 已知限制

- 碰撞体是简化几何（capsule/box），不是精细网格，不能用来做高精度碰撞检测。
- 夹爪没有做真的"力控抓取"，靠位置伺服 + 指尖摩擦；方块 0.30 kg，抓 3 kg 负载需另做。
- 桌面没有建摩擦参数调优，方块静置稳定但快速拖动会滑。
- 篮子是 5 个 box 拼的，没有做圆角/镂空；内腔 280 mm，三块 80 mm 方块按三格位可以同时放下。
  再增加方块就得换更大的筐或加隔板。
- 未接入 `pyAgxArm` 真实控制器；执行器是 MuJoCo `position` 类型，
  真实机器人是 MIT 模式（`t_ff` 全关节 ±16 N·m）。
- 深度相机是**理想深度图**：没有仿真噪声、没有边缘飞点、没有遮挡导致的空洞，
  也没有红外/结构光在黑色或高光表面的失效。测量精度 1.3–1.9 mm 是几何精度，
  不代表真机精度。要更真实可以往 `depth_cam.capture()` 里加高斯噪声与量化
  （真机 RealSense D435 在 0.5 m 处约 ±2 mm 随机误差 + 边缘 1–2 px 飞点）。
- 深度相机是**固定站位**的（三脚架），不是装在腕部。腕部相机会随夹爪动，
  带来手眼标定问题；这里刻意选了世界固定的第三人称视角，方便直接对接
  "在桌面坐标系里给出方块位姿"的下游任务。

## 排错：`pick all` 搬第二块时，第一块从篮子里消失了

**现象**：`./run.sh pick all` 依次搬三块，第一块明明已经放进篮子，
搬第二块的过程中它又回到了桌面上原来的位置。

**原因有两个，必须一起修**：

1. **`reset()` 重置了整个 qpos。**
   `ready` 关键帧的 qpos 长度等于 `m.nq`（= 30：7 个臂关节 + 2 根手指 +
   3 个 freejoint × 7），所以 `d.qpos[:] = m.key("ready").qpos`
   会把三块方块的自由关节一起写回桌面初始位姿。
   实测：手工把绿块放进篮子 `[-0.04 0.42 0.788]` 后再调一次 `reset()`，
   绿块变回 `[0.18 0 0.78]`。

2. **篮子太小，且三色共用同一个放置点。**
   第一版内腔只有 140 mm，三个 80 mm 方块并排至少要 240 mm；
   就算不重置，三块也会堆在同一点互相挤出去。

**修法**：

- `reset()` 只写机械臂（7 个关节 + 2 根手指），方块 freejoint 一律不动：

  ```python
  k = m.key("ready").qpos
  gq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in GRIP_JOINTS]          # grip_l / grip_r
  d.qpos[qad] = k[qad]                 # 只写 7 个臂关节
  d.qpos[gq] = k[gq]                   # 只写 2 根手指
  d.qvel[:] = 0
  ```

- 篮子放大到内腔 280 mm，篮内用三个 `<site>` 定义红/绿/蓝各自格位
  （三角形排布、间距 120 mm），`place` / `place_hover` / `transit`
  都按自己的格位解算。

- `--all --check` 增加两条断言，防止回归：
  每搬完一块就检查**此前已放置的方块位移 < 5 mm**；
  结束时再统一校验三块是否**同时**都在各自格位上。

关节在 qpos 里的排布（`tools/check_joints.py` 实测）：

```
joint1..joint7  qpos@0..6
grip_l          qpos@7
grip_r          qpos@8
cube_red_free   qpos@9    (+7)
cube_green_free qpos@16   (+7)
cube_blue_free  qpos@23   (+7)
nq=30  nv=27  njnt=12
```

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
