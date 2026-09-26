#!/usr/bin/env bash
# Run EVERY godmode experiment at the 24h constraint lead (ADR-0013) and 5 seeds.
# Local is truth; commit + force-push separately after this completes.
#
# Usage:  bash godmode/run_all_5seed.sh
#
# Total cost: the training steps (probes, D, MV, lambda, alpha) are the heavy
# ones; the analysis steps (beta, gamma, build) are seconds. Budget a few hours.

cd "$(dirname "$0")/.." || exit 1
P=/home/l1zle/casp-research/.venv/bin/python
L=/home/l1zle/casp-research/logs
mkdir -p "$L"
: > "$L/godmode_5seed.status"

echo "GODMODE START $(date -u +%FT%TZ)  lead=$("$P" -c 'from config import Config; print(Config().constraint_lead_hours)')" >> "$L/godmode_5seed.status"

# 1. Probes A / B / C / full fusion  (trains 4 models x 5 seeds, seed-42-era probes upgraded)
"$P" godmode/run_godmode_probes.py --config all > "$L/gm_01_probes.log" 2>&1
echo "01_probes EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 2. Move D (Winkler objective) + Move E (CQR)   (D trains 5 seeds; E reads refs)
"$P" godmode/run_godmode_de.py --moves D E > "$L/gm_02_de.log" 2>&1
echo "02_de EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 3. Move MV (level/spread factorization)         (trains 5 seeds)
"$P" godmode/run_godmode_mv_5seed.py > "$L/gm_03_mv.log" 2>&1
echo "03_mv EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 4. lambda_casf sweep 0.05 vs 0.10               (trains 5 seeds x 2 lambda)
"$P" godmode/run_godmode_lambda_5seed.py > "$L/gm_04_lambda.log" 2>&1
echo "04_lambda EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 5. Test alpha (width envelope)                  (trains 5 seeds x 3 lambda)
"$P" godmode/test_alpha.py > "$L/gm_05_alpha.log" 2>&1
echo "05_alpha EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 6. Test beta (nested conformal)                 (reads saved arrays)
"$P" godmode/test_beta.py > "$L/gm_06_beta.log" 2>&1
echo "06_beta EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 7. Test gamma (oracle routing)                  (reads saved arrays)
"$P" godmode/test_gamma.py > "$L/gm_07_gamma.log" 2>&1
echo "07_gamma EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

# 8. Aggregate verdicts (5-seed means)
"$P" godmode/godmode_build.py > "$L/gm_08_build.log" 2>&1
echo "08_build EXIT=$? $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"

echo "GODMODE_DONE $(date -u +%FT%TZ)" >> "$L/godmode_5seed.status"
