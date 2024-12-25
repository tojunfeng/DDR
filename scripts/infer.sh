version=$1
cuda_to_use=$2
topk=$3
query_file=$4
if [ -z "$1" ]; then
    version=v05191513
    cuda_to_use=4,5,6,7
    topk=200
    query_file='dataset/msmarco_bm25_official/dev_queries.tsv'
fi

echo "infer version: $version"
export CUDA_VISIBLE_DEVICES=$cuda_to_use
output_dir=./output/$version
python src/infer.py \
    --output_dir $output_dir \
    --corpus_file 'dataset/msmarco_bm25_official/passages.jsonl.gz' \
    --query_file $query_file \
    --query_length 32 \
    --passage_length 144 \
    --model_path pretrained_models/distilbert-base-uncased-TAS-B \
    --checkpoint $output_dir/checkpoint_last_epoch \
    --per_device_eval_batch_size 2048 \
    --topk $topk 2>&1 | tee $output_dir/infer.log
