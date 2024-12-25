version=$1
cuda_to_use=$2
train_file=$3
if [ -z "$1" ]; then
    version=v05192111
    cuda_to_use=6,7
    train_file=$output_dir/hn_train.jsonl
fi

echo "infer version: $version"
export CUDA_VISIBLE_DEVICES=$cuda_to_use
output_dir=./output/$version
python src/cross_encoder_infer.py \
    --output_dir $output_dir \
    --corpus_file='dataset/msmarco_bm25_official/passages.jsonl.gz' \
    --train_file=$train_file \
    --bm25_file='dataset/msmarco_bm25_official/bm25_train_score.jsonl' \
    --query_length=32 \
    --passage_length=144 \
    --model_path=pretrained_models/simlm-msmarco-reranker \
    --per_device_eval_batch_size=512 2>&1 | tee $output_dir/cross_score.log
