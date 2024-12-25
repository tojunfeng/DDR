import torch
import logging
import jsonlines
import pickle
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
from config import CrossScoreArguments
from transformers import HfArgumentParser
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encode(tokenizer: PreTrainedTokenizerFast,
           query: str, passage: str) -> BatchEncoding:
    return tokenizer(query,
                     text_pair=passage,
                     max_length=192,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')


def _worker_encode_passage(gpu_id, args):
    '''Worker function to encode passages'''
    torch.cuda.set_device(gpu_id)
    corpus = load_dataset('json', data_files=args.corpus_file)['train']
    file_to_score = load_dataset('json', data_files=args.train_file)['train']
    file_to_score = file_to_score.shard(num_shards=torch.cuda.device_count(), index=gpu_id, contiguous=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).cuda()
    model.eval()

    if os.path.exists('dataset/global_qp_score.pkl'):
        logging.info('loading cache global_qp_score.pkl...')
        with open('dataset/global_qp_score.pkl', 'rb') as file:
            global_qp_score = pickle.load(file)
    else:
        global_qp_score = {}

    def yield_passages():
        for example in file_to_score:
            query_str = example['query']
            qid = int(example['query_id'])
            for pid in example['positives']['doc_id'][:100]:
                pid = int(pid)
                passage = corpus[pid]['contents']
                title = corpus[pid]['title']
                if (qid, pid) in global_qp_score:
                    continue
                yield qid, pid, query_str, f"{title}: {passage}"
            for pid in example['negatives']['doc_id'][:100]:
                pid = int(pid)
                passage = corpus[pid]['contents']
                title = corpus[pid]['title']
                if (qid, pid) in global_qp_score:
                    continue
                yield qid, pid, query_str, f"{title}: {passage}"

    def yield_batch():
        qid_list = []
        pid_list = []
        query_list = []
        passage_list = []
        for qid, pid, query_str, passage_str in yield_passages():
            qid_list.append(qid)
            pid_list.append(pid)
            query_list.append(query_str)
            passage_list.append(passage_str)
            if len(qid_list) == args.per_device_eval_batch_size:
                yield qid_list, pid_list, encode(tokenizer, query_list, passage_list)
                qid_list = []
                pid_list = []
                query_list = []
                passage_list = []

        if len(qid_list) > 0:
            yield qid_list, pid_list, encode(tokenizer, query_list, passage_list)

    file_temp = open(f'{args.output_dir}/_temp_score_{gpu_id}.txt', 'w')
    with torch.no_grad():
        for qid_list, pid_list, batch_dict in tqdm(yield_batch(), desc=f'GPU {gpu_id}', mininterval=5, total=101*len(file_to_score)//args.per_device_eval_batch_size):
            batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
            outputs: SequenceClassifierOutput = model(**batch_dict, return_dict=True)
            for qid, pid, score in zip(qid_list, pid_list, outputs.logits):
                file_temp.write(f'{qid}\t{pid}\t{score.item()}\n')
    file_temp.close()


def encode_score(args):
    torch.multiprocessing.spawn(_worker_encode_passage, nprocs=torch.cuda.device_count(), args=(args,))
    logging.info('merging scores...')
    if os.path.exists('dataset/global_qp_score.pkl'):
        logging.info('loading cache global_qp_score.pkl...')
        with open('dataset/global_qp_score.pkl', 'rb') as file:
            global_qp_score = pickle.load(file)
    else:
        global_qp_score = {}
    # with open('dataset/global_qp_score.pkl', 'rb') as file:
    #     global_qp_score = pickle.load(file)
    old_len = len(global_qp_score)
    for gpu_id in range(torch.cuda.device_count()):
        with open(f'{args.output_dir}/_temp_score_{gpu_id}.txt', 'r') as f:
            for l in f:
                qid, pid, score = l.strip().split()
                qid = int(qid)
                pid = int(pid)
                score = float(score)
                if (qid, pid) not in global_qp_score:
                    global_qp_score[(qid, pid)] = score
    with open('dataset/global_qp_score.pkl', 'wb') as file:
        pickle.dump(global_qp_score, file)
    not_skip_num = len(global_qp_score) - old_len
    logging.info(f'added {not_skip_num} new scores, now total {len(global_qp_score)} scores')
    logging.info('writing scores...')
    original_data = []
    tot_num = 0
    with jsonlines.open(args.train_file, 'r') as f:
        for l in f:
            qid = int(l['query_id'])
            new_line = {
                'query': l['query'],
                'query_id': qid,
                'positives': {'doc_id': [int(i) for i in l['positives']['doc_id'][:100]], 'score': []},
                'negatives': {'doc_id': [int(i) for i in l['negatives']['doc_id'][:100]], 'score': []}
            }
            for pid in new_line['positives']['doc_id']:
                new_line['positives']['score'].append(global_qp_score[(qid, pid)])
                tot_num += 1
            for pid in new_line['negatives']['doc_id']:
                new_line['negatives']['score'].append(global_qp_score[(qid, pid)])
                tot_num += 1
            original_data.append(new_line)
    print(f'tot_num: {tot_num}, skip_num: {tot_num-not_skip_num}, skip rate: {(tot_num-not_skip_num)/tot_num}')
    with jsonlines.open(f"{args.output_dir}/hn_train_score.jsonl", 'w') as f:
        for l in original_data:
            f.write(l)
    logging.info('hn neagtive done!')
    
def mergeBM25HN(args):
    hnfile = f"{args.output_dir}/hn_train_score.jsonl"
    bm25file = args.bm25_file
    outputfile = f"{args.output_dir}/bm25_hn_train_score.jsonl"
    query_id2negatives = {}
    with jsonlines.open(bm25file, 'r') as f:
        for l in f:
            query_id = l['query_id']
            query_id2negatives[query_id] = l['negatives']    

    res = []
    with jsonlines.open(hnfile, 'r') as f:
        for l in f:
            query_id = l['query_id']
            if query_id in query_id2negatives:
                bm25 = query_id2negatives[query_id]
                l['negatives']['doc_id'] += bm25['doc_id']
                l['negatives']['score'] += bm25['score']
            res.append(l)
    
    with jsonlines.open(outputfile, 'w') as f:
        for l in res:
            f.write(l)

    logging.info('HN BM25 merge done!')

if __name__ == '__main__':
    set_seed(42)
    args = HfArgumentParser((CrossScoreArguments,)).parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    encode_score(args)
    mergeBM25HN(args)
    


