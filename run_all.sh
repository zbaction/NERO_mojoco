#!/usr/bin/env bash
# 从零复现：solve → build → validate → pick-place check
set -e
cd ~/NERO
PY=/home/zhangb/so101_mujoco/.venv/bin/python
echo "=== 1/4 解位姿 ==="
$PY -u solve_poses.py
echo
echo "=== 2/4 生成 XML ==="
$PY -u build_nero.py
echo
echo "=== 3/4 全量校验 ==="
$PY -u validate_nero.py 2>&1 | tail -8
echo
echo "=== 4/4 抓取-放置自检 ==="
$PY -u pick_place.py --check
