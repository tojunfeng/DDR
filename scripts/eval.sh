version=$1
output_file_name=$2
if [ -z "$1" ]; then
    version=v05191513
    output_file_name='eval.log'
fi
echo "eval version: $version"

output_dir=./output/$version
python src/metric/msmarco_eval.py dataset/msmarco_bm25_official/dev_qrels.txt $output_dir/results.top1000 2>&1 | tee $output_dir/$output_file_name
