#!/usr/bin/env bash
# 一键入口
#   ./run.sh             交互窗口（机械臂 + 桌面 + 三色方块 + 篮子）
#   ./run.sh demo        交互窗口，自动演示一遍抓取
#   ./run.sh pick        交互窗口，自动演示完整的抓取-放置（夹起绿方块放进篮子）
#   ./run.sh check       无窗口自检（抓取成功返回 0）
#   ./run.sh place       无窗口自检抓取-放置（方块落进篮子返回 0）
#   ./run.sh shots       重出成果图到 out/
#   ./run.sh detail      重出夹爪特写图到 out/
#   ./run.sh pp-shots    重出抓取-放置的 7 张过程图到 out/
#   ./run.sh validate    跑 6 段全量校验
#   ./run.sh desk        只看第一版的桌子+三色方块
set -e
cd "$(dirname "$0")"
PY=/home/zhangb/so101_mujoco/.venv/bin/python

case "${1:-}" in
  demo)     shift; exec "$PY" view_scene.py --demo "$@" ;;
  pick)     shift; exec "$PY" pick_place.py --viewer "$@" ;;
  check)    shift; exec "$PY" view_scene.py --check "$@" ;;
  place)    shift; exec "$PY" pick_place.py --check "$@" ;;
  shots)    shift; exec "$PY" render_showcase.py "$@" ;;
  detail)   shift; exec "$PY" render_detail.py "$@" ;;
  pp-shots) shift; exec "$PY" pick_place.py --shots "$@" ;;
  validate) shift; exec "$PY" validate_nero.py "$@" ;;
  desk)     shift; exec "$PY" view.py "$@" ;;
  *)        exec "$PY" view_scene.py "$@" ;;
esac
