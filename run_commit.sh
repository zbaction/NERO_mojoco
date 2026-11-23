#!/usr/bin/env bash
set -u
cd /home/zhangb/NERO || exit 1
git config user.name zbaction
git config user.email zbaction@users.noreply.github.com
git add -A
git commit -q -m "fix(pick-place): 三块方块可同时留在篮内

问题：./run.sh pick all 依次搬三块时，搬第二块会把第一块从篮子里弄回桌面。

两个根因：
1. reset() 写 d.qpos[:] = m.key('ready').qpos，而该 qpos 长度 = m.nq = 30
   （7 臂关节 + 2 指 + 3 个 freejoint x 7），于是三块方块的自由关节
   被一起重置回桌面初始位姿。改为只写臂关节与两根手指。
2. 篮子内腔只有 140 mm，三块 80 mm 方块并排至少要 240 mm，且三色
   共用同一个放置点，必然互相挤出去。改为内腔 280 mm，并用三个 site
   定义红/绿/蓝各自格位（三角形排布、间距 120 mm）。

其他改动：
- build_nero.py: BASKET_XY 改为 (-0.16, 0.42)；新增 BASKET_SLOTS 与
  slot_<col> site，作为格位的唯一数据源
- solve_poses.py: 新增 slot_xy()，place/place_hover/transit 按各自格位解算
- pick_place.py: reset() 只动机械臂；cube_in_basket() 改为跟自己的格位比；
  --all --check 增加两条断言（已放置方块位移 < 5 mm、三块同时都在篮内）
- README: 补充篮子尺寸/格位说明与本次排错记录
- 新增 tools/shot_basket_all.py 出三块合影 out/basket_three_cubes.png

验证：solve -> build -> validate 全 PASS；pick_place.py --all --check PASS
      落点偏离各自格位 2.0~2.3 mm；已放置方块最大水平扰动 0.00 mm"
echo "commit 退出码=$?"
git --no-pager log --oneline -1
git status --short | head -5
