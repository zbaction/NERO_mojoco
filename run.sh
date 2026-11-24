#!/usr/bin/env bash
# 一键入口
#   ./run.sh             交互窗口（机械臂 + 桌面 + 三色方块 + 篮子）
#   ./run.sh demo        交互窗口，自动演示一遍抓取
#   ./run.sh pick        交互窗口，自动演示完整的抓取-放置（默认绿方块）
#   ./run.sh pick red    交互窗口，搬红方块（也可 green / blue / all）
#   ./run.sh check       无窗口自检（抓取成功返回 0）
#   ./run.sh place       无窗口自检抓取-放置（方块落进篮子返回 0）
#   ./run.sh place blue  自检搬蓝方块（也可 red / all）
#   ./run.sh shots       重出成果图到 out/
#   ./run.sh detail      重出夹爪特写图到 out/
#   ./run.sh pp-shots    重出抓取-放置的过程图到 out/（带 --all 出三色 21 张）
#   ./run.sh depth       深度相机：出 out/wrist_depth_all.png 拼版（RGB/深度/点云/测量）
#   ./run.sh depth --multi  额外出 ready / green_hover / green_grasp 三个位姿的深度图
#   ./run.sh depth-check 深度相机测量自检（三块方块中心误差 < 6 mm 返回 0）
#   ./run.sh validate    跑 7 段全量校验（含深度相机）
#   ./run.sh desk        只看第一版的桌子+三色方块
#   ./run.sh live        实时监控窗口：3D 场景 + 腕部相机 RGB/深度同框
#   ./run.sh live green  打开窗口后自动搬绿方块（red / blue / all）
set -e
cd "$(dirname "$0")"
PY=/home/zhangb/so101_mujoco/.venv/bin/python

# 把第一个参数当成颜色/关键字：red|green|blue|all
pick_args() {
  case "${1:-}" in
    red|green|blue) echo --cube "$1" ;;
    all)            echo --all ;;
    *)              echo "$@" ;;
  esac
}

case "${1:-}" in
  demo)     shift; exec "$PY" view_scene.py --demo "$@" ;;
  pick)     shift; exec "$PY" pick_place.py --viewer $(pick_args "$@") ;;
  check)    shift; exec "$PY" view_scene.py --check "$@" ;;
  place)    shift; exec "$PY" pick_place.py --check $(pick_args "$@") ;;
  shots)    shift; exec "$PY" render_showcase.py "$@" ;;
  detail)   shift; exec "$PY" render_detail.py "$@" ;;
  pp-shots) shift; exec "$PY" pick_place.py --shots $(pick_args "$@") ;;
  depth)    shift; exec "$PY" depth_show.py "$@" ;;
  depth-check) shift; exec "$PY" depth_view.py --check "$@" ;;
  depth-view)  shift; exec "$PY" depth_view.py "$@" ;;
  validate) shift; exec "$PY" validate_nero.py "$@" ;;
  desk)     shift; exec "$PY" view.py "$@" ;;
  live)     shift; exec "$PY" live_monitor.py "$@" ;;
  *)        exec "$PY" view_scene.py "$@" ;;
esac
