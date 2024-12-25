export CUDA_VISIBLE_DEVICES=5
mkdir -p temp
python src/preprocess_data.py \
    --output_dir temp \
    --corpus_file dataset/msmarco_bm25_official/passages.jsonl.gz \
    --train_file dataset/msmarco_bm25_official/train.jsonl \
    --query_length=32 \
    --passage_length=144 \
    --model_path pretrained_models/simlm-msmarco-reranker \
    --per_device_eval_batch_size 512 2>&1 | tee temp/preprocess_data.log

mv temp/hn_train.jsonl dataset/msmarco_bm25_official/bm25_train_score.jsonl