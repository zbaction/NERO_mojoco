#!/usr/bin/env bash
# 端到端：solve -> build -> validate -> 三色 check（含"已放置方块不被挪动"断言）
set -u
PY=/home/zhangb/so101_mujoco/.venv/bin/python
cd /home/zhangb/NERO || exit 1
rc=0

echo "=========== 1/4 solve ==========="
$PY -u solve_poses.py || rc=1
echo "solve 退出码=$rc"

echo
echo "=========== 2/4 build ==========="
$PY -u build_nero.py || rc=1
echo "build 退出码=$rc"

echo
echo "=========== 3/4 validate ==========="
$PY -u validate_nero.py || rc=1
echo "validate 退出码=$rc"

echo
echo "=========== 4/4 pick all --check ==========="
$PY -u pick_place.py --all --check || rc=1
echo "check 退出码=$rc"

echo
echo "总退出码=$rc"
exit $rc
