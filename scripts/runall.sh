cuda_to_use=6,7
version=v2
HN_train_file_name='hn_train.jsonl'
output_dir=./output/$version

# stage 1
bash scripts/train.sh $version $cuda_to_use 'dataset/msmarco_bm25_official/bm25_train_score.jsonl'
bash scripts/infer.sh $version $cuda_to_use 1000 'dataset/msmarco_bm25_official/dev_queries.tsv'
bash scripts/eval.sh $version 'eval_stage_score_1.log'


for i in {1..6}
do
    # stage $i mine hard negatives
    rm -rf $output_dir/infer_cache/query_embeddings.npy
    bash scripts/infer.sh $version $cuda_to_use 200 'dataset/msmarco_bm25_official/train_queries.tsv'
    bash scripts/construct_train_data.sh $version $HN_train_file_name

    bash scripts/cross_encoder_score.sh $version $cuda_to_use $output_dir/hn_train.jsonl

    cp -r $output_dir/checkpoint_last_epoch $output_dir/checkpoint_last_epoch_stage_$i
    mv $output_dir/train-epoch1.log $output_dir/train-epoch1-stage_$i.log

    # stage $((i+1))
    bash scripts/train.sh $version $cuda_to_use $output_dir/bm25_hn_train_score.jsonl
    rm -rf $output_dir/infer_cache/
    bash scripts/infer.sh $version $cuda_to_use 1000 'dataset/msmarco_bm25_official/dev_queries.tsv'
    bash scripts/eval.sh $version "eval_stage_score_$((i+1)).log"
done
