#!/usr/bin/env bash
# 篮子几何改了 → 必须 build → solve → build，否则 site 不存在 / 关键帧是旧值
set -u
PY=/home/zhangb/so101_mujoco/.venv/bin/python
cd /home/zhangb/NERO || exit 1

echo "=========== 1/3 build（先建带 site 的新场景） ==========="
$PY -u build_nero.py || { echo "build 失败"; exit 1; }

echo
echo "=========== 2/3 solve（按新 site 解 19 个位姿） ==========="
$PY -u solve_poses.py || { echo "solve 失败"; exit 1; }

echo
echo "=========== 3/3 build（把新关键帧写回 XML） ==========="
$PY -u build_nero.py || { echo "build 失败"; exit 1; }

echo
echo "完成。"
