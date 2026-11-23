# NERO 机械臂 MuJoCo 场景

AgileX **NERO** 七自由度机械臂 + 两指夹爪，装在木桌上，桌面上摆着红/绿/蓝三个方块。
模型由官方 URDF/质量表逐项转换为 MuJoCo MJCF，机械臂可以真的走过去**夹起方块并抬离桌面**。

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
./run.sh check        # 无窗口自检，抓取成功返回 0
./run.sh validate     # 6 段全量校验
./run.sh shots        # 重出成果图到 out/
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

`` poses.json `` 里的 5 个位姿全部 **位置误差 0.00 mm、倾角 0.00°、偏航误差 0.00°**：

| 关键帧 | 关节角 (°) | 余量 |
| --- | --- | --- |
| ready | 19.93, -54.13, 83.01, 34.64, -17.40, -32.56, -22.10 | 11.4° |
| hover | 34.22, -68.41, 88.62, 71.10, -20.15, -8.07, -46.58 | 33.6° |
| grasp | 34.69, -75.75, 102.85, 78.93, -16.68, 9.45, -49.24 | 26.3° |
| grasp_red | -2.21, -82.27, 96.55, 61.83, -9.91, 2.06, -51.41 | 19.7° |
| grasp_blue | 61.15, -86.45, 88.86, 61.00, -2.56, -2.71, 20.06 | 15.5° |

算法是"多起点随机射击 + 软限位阻尼最小二乘"：随机撒 500 组初始关节角，
每组迭代 320 次，取残差最小的解，再在候选里按余量 / 关节位移量打分。

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
| 运动学 | 5 个关键帧指尖中点精确命中目标，工具 z 轴全为 (0,0,-1) |
| 稳定性 | ready 保持 3 s，最大漂移 0.318° |
| 抓取 | hover → grasp（偏差 3.5 mm）→ 4 个 pad↔cube 接触 → 抬起 **Δz = +0.117 m** |
| 渲染 | 机械臂在 4 个机位的画面占比 1.55% / 4.07% / 4.51% / 2.59% |

判定不靠肉眼：像素占比用"关掉 geomgroup[2] 前后两次渲染差分"客观测量。

## 文件

| 文件 | 说明 |
| --- | --- |
| `build_nero.py` | **生成器**：读 URDF + CSV + poses.json，写出 `nero_arm.xml` 与 `nero_table_scene.xml` |
| `nero_table_scene.xml` | 生成物：桌子 + 三色方块 + NERO 机械臂（主场景） |
| `nero_arm.xml` | 生成物：只有机械臂，基座在原点（便于单独复用） |
| `solve_poses.py` | 6 残差阻尼最小二乘 IK，写 `poses.json` |
| `poses.json` | 5 个关键帧的关节角 |
| `validate_nero.py` | 6 段全量校验 |
| `render_showcase.py` | 成果图（自动取景） |
| `render_detail.py` | 夹爪抓取特写 |
| `view_scene.py` | 交互窗口 / `--demo` 演示 / `--check` 自检 |
| `view.py` | 第一版：只有桌子 + 三色方块 |
| `check_colors.py` | 校验渲染图里确实有三种不同颜色 |
| `run.sh` | 一键入口 |
| `assets/nero/` | STL 网格 + 原始 CSV |
| `out/` | 渲染结果 |
| `tools/` | 一次性诊断脚本（FK 交叉验证、抓取诊断等） |

## 复现

```bash
cd ~/NERO
/home/zhangb/so101_mujoco/.venv/bin/python solve_poses.py   # 重新解位姿 → poses.json
/home/zhangb/so101_mujoco/.venv/bin/python build_nero.py    # 重新生成 XML
/home/zhangb/so101_mujoco/.venv/bin/python validate_nero.py # 校验
```

## 已知限制

- 碰撞体是简化几何（capsule/box），不是精细网格，不能用来做高精度碰撞检测。
- 夹爪没有做真的"力控抓取"，靠位置伺服 + 指尖摩擦；方块 0.30 kg，抓 3 kg 负载需另做。
- 桌面没有建摩擦参数调优，方块静置稳定但快速拖动会滑。
- 未接入 `pyAgxArm` 真实控制器；执行器是 MuJoCo `position` 类型，
  真实机器人是 MIT 模式（`t_ff` 全关节 ±16 N·m）。
