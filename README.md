# NERO_mojoco — MuJoCo 桌面 + 三个彩色方块

WSL 路径：`~/NERO`（Windows 侧：`\\wsl.localhost\Ubuntu-22.04\home\zhangb\NERO`）

## 文件

| 文件 | 说明 |
| --- | --- |
| `desk_three_cubes.xml` | MJCF 模型：一张四条腿的桌子 + 桌面上红/绿/蓝三个 10cm 方块 |
| `view.py` | 打开交互窗口 / 离屏渲染 |
| `check_colors.py` | 自动校验渲染图里确实有三种不同颜色（不靠肉眼） |
| `out/` | 离屏渲染出的三张预览图（overview / front / top） |
| `run.sh` | 一键运行 |

## 运行

```bash
cd ~/NERO
./run.sh              # 打开交互窗口（走 WSLg 显示到 Windows 桌面）
./run.sh --render     # 只渲染 png 到 out/
```

用的是已经装好 mujoco 3.17 的解释器：`/home/zhangb/so101_mujoco/.venv/bin/python`
（系统 `python3` 里没有 mujoco）。

## 模型要点

- 单位米。地面在 z=0，桌面 0.8m × 1.2m，台面上表面 z = 0.74。
- 方块边长 0.1m，沿 y 方向排在台面中间，各自带一个 `freejoint`，
  所以它们是真正参与物理的刚体，不是贴图。
- 方块初始各自带一点 z 轴旋转（12° / -8° / 20°），看起来不是死板对齐。
- `timestep=0.002`、`integrator=implicitfast`、`cone=elliptic`、`impratio=10`，
  三个方块放上去后 5 秒内位移 < 0.1mm，能稳稳静止。
- 预设相机：`overview`、`front`、`top`，在交互窗口里可以按 Tab 切换/自由视角。

## 自定义

- 改颜色：`<material name="mat_red" rgba="..."/>`
- 改位置：`<body name="cube_red" pos="x y z">`
- 改大小：`<geom size="0.05 0.05 0.05"/>` 三个数是半长（半边长），
  同时要把 `pos` 的 z 调成 `0.74 + 半边长`。
- 加方块：复制一个 `<body>` + `<freejoint>` + `<geom>` 块即可。