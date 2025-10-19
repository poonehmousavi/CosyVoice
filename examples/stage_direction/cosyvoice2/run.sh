#!/usr/bin/env bash
set -euo pipefail
. ./path1.sh || exit 1;

# --- Resolve important paths ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"          # /code/examples/examples/stage_direction/cosyvoice2
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd -P)"                    # /code  (double examples -> 4 levels up)

# --- Config you said you have ---
DATA_ROOT="/data/rita/RitaGearspark"                                     # audio + metadata live here
LIST_FILE="$DATA_ROOT/metadata.txt"                                      # file_name|transcript|...
DES_BASE="$SCRIPT_DIR/data/stage_direction"                              # outputs go next to this script
PRETRAINED_DIR="/code/pretrained_models/CosyVoice2-0.5B"            # adjust if your models are elsewhere

# Tools (absolute paths in repo)
EXTRACT_EMB="/code/tools/extract_embedding.py"
EXTRACT_TOK="/code/tools/extract_speech_token.py"
MAKE_PARQUET="/code/tools/make_parquet_list.py"

# Params
NUM_PROCESSES=10
NUM_UTTS_PER_PARQUET=1000
stage="${stage:-4}"
stop_stage="${stop_stage:-5}"

# --- Sanity checks ---
echo "SCRIPT_DIR     : $SCRIPT_DIR"
echo "REPO_ROOT      : $REPO_ROOT"
echo "DATA_ROOT      : $DATA_ROOT"
echo "LIST_FILE      : $LIST_FILE"
echo "DES_BASE       : $DES_BASE"
echo "PRETRAINED_DIR : $PRETRAINED_DIR"

[[ -d "$DATA_ROOT" ]]  || { echo "ERROR: DATA_ROOT not found: $DATA_ROOT"; exit 1; }
[[ -f "$LIST_FILE" ]]  || { echo "ERROR: LIST_FILE not found: $LIST_FILE"; exit 1; }

# Pick audio_folder automatically based on first column format
FIRST_COL="$(grep -v '^[[:space:]]*#' "$LIST_FILE" | sed -n '1p' | cut -d'|' -f1)"
# If the first column already contains a subpath like "rita/RitaGearspark/...":
if [[ "$FIRST_COL" == */* ]]; then
  AUDIO_FOLDER="/data"                              # so /data + "rita/RitaGearspark/xxx" -> /data/rita/RitaGearspark/xxx
else
  AUDIO_FOLDER="/data/rita/RitaGearspark"          # so /data/rita/RitaGearspark + "xxx"   -> /data/rita/RitaGearspark/xxx
fi
echo "AUTO audio_folder: $AUDIO_FOLDER (first_col='$FIRST_COL')"

# Check pretrained ONNX paths (adjust if you keep them elsewhere)
CAMP_ONNX="$PRETRAINED_DIR/campplus.onnx"
TOK_ONNX="$PRETRAINED_DIR/speech_tokenizer_v2.onnx"
[[ -f "$EXTRACT_EMB" ]] || { echo "ERROR: Missing $EXTRACT_EMB"; exit 1; }
[[ -f "$EXTRACT_TOK" ]] || { echo "ERROR: Missing $EXTRACT_TOK"; exit 1; }
[[ -f "$MAKE_PARQUET" ]]|| { echo "ERROR: Missing $MAKE_PARQUET"; exit 1; }
[[ -f "$CAMP_ONNX" ]]   || { echo "ERROR: Missing $CAMP_ONNX"; exit 1; }
[[ -f "$TOK_ONNX" ]]    || { echo "ERROR: Missing $TOK_ONNX"; exit 1; }

mkdir -p "$DES_BASE"

# -------------------- Stage 0: data prep --------------------
if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
  echo "Stage 0: Data preparation (wav.scp/text/utt2spk/spk2utt)"
  python -u "$SCRIPT_DIR/prepare_data.py" \
    --list_file "$LIST_FILE" \
    --audio_folder "$AUDIO_FOLDER" \
    --des_dir "$DES_BASE" \
    --seed 42
fi

# -------------------- Stage 1: speaker embeddings --------------------
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  echo "Stage 1: Extract CampPlus speaker embeddings"
  for x in train dev; do
    in_dir="$DES_BASE/$x"
    [[ -d "$in_dir" ]] || { echo "WARN: Missing split dir $in_dir, skipping"; continue; }
    python -u "$EXTRACT_EMB" --dir "$in_dir" --onnx_path "$CAMP_ONNX"
  done
fi

# -------------------- Stage 2: speech tokens --------------------
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
  echo "Stage 2: Extract discrete speech tokens"
  for x in train dev; do
    in_dir="$DES_BASE/$x"
    [[ -d "$in_dir" ]] || { echo "WARN: Missing split dir $in_dir, skipping"; continue; }
    python -u "$EXTRACT_TOK" --dir "$in_dir" --onnx_path "$TOK_ONNX"
  done
fi

# -------------------- Stage 3: parquet lists --------------------
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
  echo "Stage 3: Prepare parquet format data"
  for x in train dev; do
    in_dir="$DES_BASE/$x"
    out_dir="$DES_BASE/$x/parquet"
    [[ -d "$in_dir" ]] || { echo "WARN: Missing split dir $in_dir, skipping"; continue; }
    mkdir -p "$out_dir"
    python -u "$MAKE_PARQUET" \
      --num_utts_per_parquet "$NUM_UTTS_PER_PARQUET" \
      --num_processes "$NUM_PROCESSES" \
      --src_dir "$in_dir" \
      --des_dir "$out_dir"
  done
fi


# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $DES_BASE/train/parquet/data.list > $DES_BASE/train.data.list
  cat $DES_BASE/dev/parquet/data.list > $DES_BASE/dev.data.list
  # NOTE will update llm/hift training later
  for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      /code/cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config $SCRIPT_DIR/conf/cosyvoice2.yaml \
      --train_data $DES_BASE/train.data.list \
      --cv_data $DES_BASE/dev.data.list \
      --qwen_pretrain_path $PRETRAINED_DIR/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $$PRETRAINED_DIR/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config $SCRIPT_DIR/conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python /code/cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi