version=$1
output_train_file=$2
if [ -z "$1" ]; then
    version=v05191219
    output_train_file=hn_train.jsonl
fi
output_dir=./output/$version
echo $output_train_file
python src/construct_train_data.py \
    --query_file='dataset/msmarco_bm25_official/train_queries.tsv' \
    --qrel_file='dataset/msmarco_bm25_official/train_qrels.txt' \
    --output_file=$output_dir/$output_train_file \
    --topk_file=$output_dir/results.top200 \
    --depth=100 \
    --seed=42 2>&1 | tee $output_dir/construct_train_data.log

