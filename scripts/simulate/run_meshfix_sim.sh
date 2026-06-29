#!/bin/bash
# 干净模型(meshfix, V-HACD 修复几何 + 重标数据训)部署仿真。
# 部署参数已按碰撞率评测调好:thr8(meshfix gamma 尺度下移, 必须 ≥8 才够 braking margin)、
# ctrl-dt 1ms(20ms 会 swing-in 自碰)、sca-eps 0.1。在 thr8 下 ~0% 碰撞。
# 用法:
#   bash scripts/simulate/run_meshfix_sim.sh                       # GUI 看一遍 (seed28)
#   bash scripts/simulate/run_meshfix_sim.sh --rand-init --seed 3  # 随机初始, 指定种子
#   bash scripts/simulate/run_meshfix_sim.sh --no-gui --record out.mp4 --rand-init --seed 3  # 无显示器, 出视频
# 任何额外 flag 直接透传给 simulate_dual_openarm_cls.py
cd /home/yicong/openarm_test
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 scripts/simulate/simulate_dual_openarm_cls.py \
  --model-path assets/models/gamma_cls_meshfix_d128.pt \
  --gamma-threshold 8 \
  --sca-eps 0.1 \
  --ctrl-dt 0.001 \
  --duration 20 \
  "$@"
