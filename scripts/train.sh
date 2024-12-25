version=$1
cuda_to_use=$2
train_file=$3
if [ -z "$1" ]; then
    version=v05090952
    cuda_to_use=6,7
    train_file='dataset/msmarco_distillation/kd_train.jsonl'
fi
echo "train version: $version"

export CUDA_HOME=/usr/local/cuda
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
output_dir=./output/$version
epochs=1
mkdir -p $output_dir

# Train
deepspeed --include localhost:$cuda_to_use --master_port 45179 src/train.py --deepspeed ds_config.json \
    --do_train \
    --do_kd_loss \
    --bf16 1\
    --output_dir $output_dir \
    --model_path pretrained_models/distilbert-base-uncased-TAS-B \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs $epochs \
    --learning_rate 2e-5 \
    --query_length 32 \
    --passage_length 144 \
    --num_negatives 200 \
    --evaluation_strategy no \
    --checkpoint $output_dir/checkpoint_last_epoch \
    --corpus_file 'dataset/msmarco_bm25_official/passages.jsonl.gz' \
    --train_file $train_file \
    --dev_file 'dataset/msmarco_bm25_official/dev.jsonl' \
    --message "distilbert-base-uncased msmarco_bm25_official" \
    --seed 42 2>&1 | tee $output_dir/train-epoch${epochs}.log

