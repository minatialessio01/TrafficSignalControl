#!/bin/bash
set -u

CONFIGS=(
  configs/config_4x4_100m_train1.json
  configs/config_4x4_100m_train2.json
  configs/config_4x4_100m_train3.json
  configs/config_4x4_100m_6k_peak.json
  configs/config_4x4_100m_6k_peak2.json
  configs/config_4x4_100m_6k_peak3.json
  configs/config_4x4_100m_6k_peak4.json
  configs/config_4x4_100m_6k_peak5.json
  configs/config_4x4_200m_6k_flat.json
  configs/config_4x4_200m_6k_flat2.json
  configs/config_4x4_200m_6k_flat3.json
  configs/config_4x4_200m_6k_flat4.json
  configs/config_4x4_200m_6k_flat5.json
  configs/config_4x4_200m_6k_peak.json
  configs/config_4x4_200m_6k_peak2.json
  configs/config_4x4_200m_6k_peak3.json
  configs/config_4x4_200m_6k_peak4.json
  configs/config_4x4_200m_6k_peak5.json
  configs/config_5x5_100m_9.4k_flat.json
  configs/config_5x5_100m_9.4k_flat2.json
  configs/config_5x5_100m_9.4k_flat3.json
  configs/config_5x5_100m_9.4k_flat4.json
  configs/config_5x5_100m_9.4k_flat5.json
  configs/config_5x5_100m_9.4k_peak.json
  configs/config_5x5_100m_9.4k_peak2.json
  configs/config_5x5_100m_9.4k_peak3.json
  configs/config_5x5_100m_9.4k_peak4.json
  configs/config_5x5_100m_9.4k_peak5.json
  configs/config_6x6_100m_11.5k_flat.json
  configs/config_6x6_100m_11.5k_flat2.json
  configs/config_6x6_100m_11.5k_flat3.json
  configs/config_6x6_100m_11.5k_flat4.json
  configs/config_6x6_100m_11.5k_flat5.json
  configs/config_6x6_100m_11.5k_peak.json
  configs/config_6x6_100m_11.5k_peak2.json
  configs/config_6x6_100m_11.5k_peak3.json
  configs/config_6x6_100m_11.5k_peak4.json
  configs/config_6x6_100m_11.5k_peak5.json
)

OUT="results/maxpressure_masked"
TOTAL=${#CONFIGS[@]}
DONE=0
echo "=== INIZIO test maxpressure_masked: $TOTAL config ==="

for cfg in "${CONFIGS[@]}"; do
  docker run --rm -v "$(pwd)":/workspace -w /workspace metastgat:latest \
    python3 -u scripts/test.py \
      --config "$cfg" \
      --model MaxPressure \
      --model-id maxpressure_masked \
      --ablation pro \
      --mp-respect-mask \
      --output-dir "$OUT" \
      --n-eval 1 > /tmp/_test_mp_masked_last.txt 2>&1
  RC=$?
  DONE=$((DONE + 1))
  if [ $RC -ne 0 ]; then
    echo "[ERRORE] su $cfg (exit $RC):"
    tail -20 /tmp/_test_mp_masked_last.txt
  fi
  if [ $((DONE % 10)) -eq 0 ]; then
    echo "[PROGRESS] $DONE/$TOTAL completati"
  fi
done

echo "=== FINE test maxpressure_masked: $DONE/$TOTAL run completati ==="
