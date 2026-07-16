#!/bin/bash
#
# Reproduces the ARIANE ablation runs (baseline, enhanced, LLM-guided)
# across increasing net counts. Expects ariane.cap and ariane.net under data/.
#
# Usage: ./run_experiments.sh
set -euo pipefail

CAP="data/ariane.cap"
NET="data/ariane.net"
OUT="artifacts"

mkdir -p "$OUT"

echo "Running AlphaRoute experiments..."
for nets in 100 200 400 500; do
    echo "  [LLM-guided]  max_nets=$nets"
    python router.py -net "$NET" -cap "$CAP" -output "$OUT/ariane_llm_$nets.route" \
        -max_nets "$nets" -enhanced -llm > "$OUT/ariane_llm_$nets.log" 2>&1

    echo "  [Enhanced]    max_nets=$nets"
    python router.py -net "$NET" -cap "$CAP" -output "$OUT/ariane_enh_$nets.route" \
        -max_nets "$nets" -enhanced > "$OUT/ariane_enh_$nets.log" 2>&1

    echo "  [Baseline]    max_nets=$nets"
    python router.py -net "$NET" -cap "$CAP" -output "$OUT/ariane_base_$nets.route" \
        -max_nets "$nets" > "$OUT/ariane_base_$nets.log" 2>&1
done
echo "Done."
