#!/bin/bash
set -e

# =============================================================
# VideoGPA Replication Script — Generation + Scoring
# Usage: bash replicate.sh [options]
#
# Options:
#   --mode          dpo | sft | original          (default: dpo)
#   --lora_path     path to LoRA weights           (default: checkpoints/VideoGPA-I2V-lora)
#   --output_dir    where to save videos           (default: output/replicate)
#   --prompt_json   path to captions JSON          (default: dl3dv_video_captions/captions_1K.json)
#   --dl3dv_dir     path to DL3DV-10K root         (default: /datasets/DL3DV-10K)
#   --devices       comma-separated GPU ids        (default: 0)
#   --num_prompts   number of prompts to use       (default: 100)
#   --seeds         comma-separated seeds          (default: 456)
#   --num_frames    frames sampled per video       (default: 10)
#   --skip_gen      skip generation, only score    (default: false)
#   --skip_score    skip scoring, only generate    (default: false)
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
MODE="dpo"
LORA_PATH="${SCRIPT_DIR}/checkpoints/VideoGPA-I2V-lora"
OUTPUT_DIR="${SCRIPT_DIR}/output/replicate"
PROMPT_JSON="${SCRIPT_DIR}/dl3dv_video_captions/captions_1K.json"
DL3DV_DIR="/datasets/DL3DV-10K"
DEVICES="0"
NUM_PROMPTS="100"
SEEDS="456"
NUM_FRAMES="10"
SKIP_GEN=false
SKIP_SCORE=false

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2";        shift 2 ;;
        --lora_path)   LORA_PATH="$2";   shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --prompt_json) PROMPT_JSON="$2"; shift 2 ;;
        --dl3dv_dir)   DL3DV_DIR="$2";   shift 2 ;;
        --devices)     DEVICES="$2";     shift 2 ;;
        --num_prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --seeds)       SEEDS="$2";       shift 2 ;;
        --num_frames)  NUM_FRAMES="$2";  shift 2 ;;
        --skip_gen)    SKIP_GEN=true;    shift ;;
        --skip_score)  SKIP_SCORE=true;  shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCORE_DIR="${OUTPUT_DIR}"
SCORE_CSV="${OUTPUT_DIR}/scores.csv"
SCORE_JSON="${OUTPUT_DIR}/scores.json"

echo "========================================="
echo "  VideoGPA Replication"
echo "========================================="
echo "  mode        : $MODE"
echo "  lora_path   : $LORA_PATH"
echo "  output_dir  : $OUTPUT_DIR"
echo "  prompt_json : $PROMPT_JSON"
echo "  dl3dv_dir   : $DL3DV_DIR"
echo "  devices     : $DEVICES"
echo "  num_prompts : $NUM_PROMPTS"
echo "  seeds       : $SEEDS"
echo "  num_frames  : $NUM_FRAMES"
echo "========================================="

# ---------- Step 1: Generate ----------
if [ "$SKIP_GEN" = false ]; then
    echo ""
    echo "Step 1: Generating videos..."
    RUN_MODE="$MODE" \
    RUN_LORA_PATH="$LORA_PATH" \
    RUN_OUTPUT_DIR="$OUTPUT_DIR" \
    PROMPT_JSON="$PROMPT_JSON" \
    DL3DV_BASE_DIR="$DL3DV_DIR" \
    RUN_DEVICES="$DEVICES" \
    RUN_NUM_PROMPTS="$NUM_PROMPTS" \
    RUN_SEEDS="$SEEDS" \
    python "${SCRIPT_DIR}/replicate.py"
    echo "Generation done. Videos saved to: $OUTPUT_DIR"
fi

# ---------- Step 2: Score ----------
if [ "$SKIP_SCORE" = false ]; then
    echo ""
    echo "Step 2: Scoring videos..."
    SCORE_DEVICES="$DEVICES" \
    SCORE_BASE_DIR="$SCORE_DIR" \
    SCORE_BACKBONE="da3" \
    SCORE_NUM_FRAMES="$NUM_FRAMES" \
    SCORE_SEED_FILTER="$(echo "$SEEDS" | tr ',' '\n' | tail -1)" \
    SCORE_OUTPUT_CSV="$SCORE_CSV" \
    SCORE_OUTPUT_JSON="$SCORE_JSON" \
    python "${SCRIPT_DIR}/replicate_scorer.py"
    echo "Scoring done."
    echo "  CSV : $SCORE_CSV"
    echo "  JSON: $SCORE_JSON"
fi

echo ""
echo "All done."
