#!/usr/bin/env bash
# 快速看分类器 barrier 效果。
# 用法:  bash scripts/simulate/run_demo.sh [threshold] [model] [额外参数...]
#   bash scripts/simulate/run_demo.sh            # home, thr5, GUI
#   bash scripts/simulate/run_demo.sh 8          # 换阈值
#   bash scripts/simulate/run_demo.sh 5 assets/models/gamma_cls_clean_d128_120ep.pt
#   bash scripts/simulate/run_demo.sh 5 "" --rand-init --seed 7 --rand-scale 0.4
#   bash scripts/simulate/run_demo.sh 5 "" --no-gui --record output/demo.mp4 --record-every 1
cd "$(dirname "$0")/../.." || exit 1            # -> 项目根目录
THR="${1:-5}"
MODEL="${2:-assets/models/gamma_cls_clean_d128.pt}"
[ -n "$2" ] && shift 2 || shift 1               # 余下都是额外参数
python3 scripts/simulate/simulate_dual_openarm_cls.py \
  --model-path "$MODEL" --gamma-threshold "$THR" "$@"
