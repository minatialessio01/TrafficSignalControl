#!/bin/bash

CONFIGS=(
  configs/config_4x4_100m_6k_peak2.json
  configs/config_4x4_100m_6k_peak3.json
  configs/config_4x4_200m_6k_flat2.json
  configs/config_4x4_200m_6k_flat3.json
  configs/config_4x4_200m_6k_peak2.json
  configs/config_4x4_200m_6k_peak3.json
  configs/config_5x5_100m_9.4k_flat2.json
  configs/config_5x5_100m_9.4k_flat3.json
  configs/config_5x5_100m_9.4k_peak2.json
  configs/config_5x5_100m_9.4k_peak3.json
  configs/config_6x6_100m_11.5k_flat2.json
  configs/config_6x6_100m_11.5k_flat3.json
  configs/config_6x6_100m_11.5k_peak2.json
  configs/config_6x6_100m_11.5k_peak3.json
)

# model_id|model_type|ablation|alpha|reward_mode|num_layers|sonar_recurrences
MODELS=(
  "ablation_environment_official|MetaSTGAT|environment|0.5|paper|1|2"
  "ablation_replay_stability_official|MetaSTGAT|replay_stability|0.08|custom|1|2"
  "ablation_replay_stability_official_150ep|MetaSTGAT|replay_stability|0.08|custom|1|2"
  "ablation_rl_core_official|MetaSTGAT|rl_core|0.08|custom|1|2"
  "ablation_temporal_official|MetaSTGAT|temporal|0.08|custom|1|2"
  "metastgat_2l_pro_0.08_official|MetaSTGAT|pro|0.08|custom|2|2"
  "metastgat_paper_official|MetaSTGAT|paper|0.5|paper|1|2"
  "metastgat_pro_0.00_official|MetaSTGAT|pro|0.0|custom|1|2"
  "metastgat_pro_0.02_official|MetaSTGAT|pro|0.02|custom|1|2"
  "metastgat_pro_0.04_official|MetaSTGAT|pro|0.04|custom|1|2"
  "metastgat_pro_0.08_official|MetaSTGAT|pro|0.08|custom|1|2"
  "metastgat_pro_0.08_official_150ep|MetaSTGAT|pro|0.08|custom|1|2"
  "metastgcn_1l_pro_0.08_official|MetaSTGNN|pro|0.08|custom|1|2"
  "metastgcn_2l_pro_0.08_official|MetaSTGNN|pro|0.08|custom|2|2"
  "metastsonar_l2_pro_0.08_official|MetaSTSONAR|pro|0.08|custom|1|2"
  "metastsonar_l2_pro_0.08_official_150ep|MetaSTSONAR|pro|0.08|custom|1|2"
  "metastsonar_l4_pro_0.08_official|MetaSTSONAR|pro|0.08|custom|1|4"
  "fixedtime|FixedTime|pro|0.5|custom|1|2"
  "maxpressure|MaxPressure|pro|0.5|custom|1|2"
)

TOTAL=0
DONE=0
for entry in "${MODELS[@]}"; do
  TOTAL=$((TOTAL + ${#CONFIGS[@]}))
done
echo "=== INIZIO: ${#MODELS[@]} modelli x ${#CONFIGS[@]} config = $TOTAL run ==="

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MID MTYPE ABLATION ALPHA RMODE NLAYERS SREC <<< "$entry"
  OUT="results/$MID"
  for cfg in "${CONFIGS[@]}"; do
    docker run --rm -v "$(pwd)":/workspace -w /workspace metastgat:latest \
      python3 -u scripts/test.py \
        --config "$cfg" \
        --model "$MTYPE" \
        --model-id "$MID" \
        --ablation "$ABLATION" \
        --reward-mode "$RMODE" \
        --alpha "$ALPHA" \
        --num-layers "$NLAYERS" \
        --sonar-recurrences "$SREC" \
        --output-dir "$OUT" \
        --n-eval 1 > /tmp/_last_test_out.txt 2>&1
    RC=$?
    DONE=$((DONE + 1))
    if [ $RC -ne 0 ]; then
      echo "[ERRORE] $MID su $cfg (exit $RC):"
      tail -20 /tmp/_last_test_out.txt
    fi
    if [ $((DONE % 20)) -eq 0 ]; then
      echo "[PROGRESS] $DONE/$TOTAL completati"
    fi
  done
  echo "[MODELLO COMPLETATO] $MID (${#CONFIGS[@]} nuove config testate)"
done

echo "=== FINE: $DONE/$TOTAL run completati ==="
