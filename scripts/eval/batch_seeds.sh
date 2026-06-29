#!/usr/bin/env bash
# Batch INTER-ARM avoidance success over seeds.
# Fail = an inter-arm COLLISION occurred (sim prints "COLLISION" and breaks).
# OK   = ran without inter-arm collision; report min true inter-arm distance.
# Usage: batch_seeds.sh <tag> <extra sim args...>     (SEEDS env overrides list)
set -u
cd /home/yicong/openarm_test
TAG="${1:-run}"; shift || true
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
ok=0; tot=0; fails=""
for s in $SEEDS; do
  out=$(timeout 300 python3 scripts/simulate/simulate_dual_openarm_cls.py \
          --no-gui --rand-init --seed "$s" --duration 20 "$@" 2>/dev/null)
  tot=$((tot+1))
  if echo "$out" | grep -q "COLLISION"; then
    im=$(echo "$out" | grep -m1 "COLLISION" | grep -oE 'inter_arm=[-+][0-9.]+' | sed 's/inter_arm=//')
    printf "  seed %3d : COLLIDE inter_arm=%smm\n" "$s" "${im:-?}"; fails="$fails $s"
  else
    d=$(echo "$out" | grep -m1 'min true d' | grep -oE '[-+][0-9]+\.[0-9]+mm' | head -1 | sed 's/mm//')
    if [ -z "$d" ]; then
      n=$(echo "$out" | grep -oE 'Steps +: [0-9]+' | grep -oE '[0-9]+')
      printf "  seed %3d : SHORT(%s steps,no-collide)\n" "$s" "${n:-?}"; ok=$((ok+1))
    else
      ok=$((ok+1)); printf "  seed %3d : OK      min=%smm\n" "$s" "$d"
    fi
  fi
done
echo "===== [$TAG] 无臂间碰撞 $ok/$tot  碰撞种子:${fails:- 无} ====="
