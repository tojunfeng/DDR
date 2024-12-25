import torch
import numpy as np
import pandas as pd
import logging
import os
import torch.multiprocessing.spawn
import faiss
from transformers import AutoTokenizer, DataCollatorWithPadding, HfArgumentParser
from transformers.utils import PaddingStrategy
from datasets import load_dataset, Dataset
from model import BiencodeModel
from functools import partial
from tqdm import tqdm
from config import InferArguments

def _passage_transform(tokenizer, args ,examples):
    '''Transform passages'''
    batch_input = tokenizer(
        examples['title'],
        text_pair=examples['contents'],
        padding=PaddingStrategy.DO_NOT_PAD,
        max_length=args.passage_length,
        truncation=True
    )
    return batch_input

def _query_transform(tokenizer, args ,examples):
    '''Transform passages'''
    batch_input = tokenizer(
        examples['query'],
        padding=PaddingStrategy.DO_NOT_PAD,
        max_length=args.query_length,
        truncation=True
    )
    return batch_input


@torch.no_grad()
def _worker_encode_passage(gpu_id, args):
    '''Worker function to encode passages'''
    # Load model
    torch.cuda.set_device(gpu_id)
    output_path = os.path.join(args.output_dir, f'infer_cache')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    output_path = os.path.join(output_path, f'passage_embeddings_{gpu_id}.npy')
    if os.path.exists(output_path):
        logging.info(f'GPU {gpu_id} has already encoded passages, skip...')
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = BiencodeModel.build(args, from_checkpoint=True)
    model.cuda()
    model.eval()

    # Load dataset
    dataset = load_dataset('json', data_files=args.corpus_file)
    dataset = dataset['train']
    if args.do_data_sample:
        dataset = dataset.select(range(50000))
    dataset = dataset.shard(num_shards=torch.cuda.device_count(), index=gpu_id, contiguous=True)
    dataset.set_transform(partial(_passage_transform, tokenizer, args))
    data_collator = DataCollatorWithPadding(tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
        pin_memory=True
    )
    logging.info(f'GPU {gpu_id} has {len(data_loader)} batches with {len(dataset)} passages')

    # Encode passages
    all_embeddings = []
    for batch in tqdm(data_loader, mininterval=50, desc=f'encoding passages with GPU {gpu_id}'):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        p_embedding = model.p_model(batch)['p_embedding']
        all_embeddings.append(p_embedding.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Save embeddings
    np.save(output_path, all_embeddings)


def encode_passages(args):
    '''Encode passages'''
    logging.info(f'Encoding passages with {torch.cuda.device_count()} GPUs...')
    output_path = os.path.join(args.output_dir, f'infer_cache')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.multiprocessing.spawn(_worker_encode_passage, nprocs=torch.cuda.device_count(), args=(args,))

def encode_queries(args):
    '''Encode queries'''
    logging.info(f'Encoding queries')
    output_path = os.path.join(args.output_dir, f'infer_cache')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, f'query_embeddings.npy')
    if os.path.exists(output_path):
        logging.info(f'Queries have already been encoded, skip...')
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = BiencodeModel.build(args, from_checkpoint=True)
    model.cuda()
    model.eval()
    df = pd.read_csv(args.query_file, sep='\t', header=None)
    df.rename(columns={0: 'query_id', 1: 'query'}, inplace=True)
    dataset = Dataset.from_pandas(df)
    if args.do_data_sample and len(dataset) > 10000:
        dataset = dataset.select(range(10000))
    # print(dataset)
    dataset.set_transform(partial(_query_transform, tokenizer, args))
    data_collator = DataCollatorWithPadding(tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
        pin_memory=True
    )
    all_embeddings = []
    for batch in tqdm(data_loader, mininterval=50, desc=f'encoding queries'):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            q_embedding = model.q_model(batch)['q_embedding']
        all_embeddings.append(q_embedding.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_path, all_embeddings)
    
    del model
    torch.cuda.empty_cache()


def _worker_search(gpu_id, args):
    '''Worker function to search'''
    torch.cuda.set_device(gpu_id)
    output_path = os.path.join(args.output_dir, f'infer_cache')
    query_embeddings = np.load(os.path.join(output_path, 'query_embeddings.npy'))
    qid_list = []
    with open(args.query_file, 'r') as f:
        for line in f:
            qid = line.strip().split('\t')[0]
            qid_list.append(int(qid))

    query_size = query_embeddings.shape[0] // torch.cuda.device_count() + 1
    qid_list = qid_list[gpu_id*query_size:(gpu_id+1)*query_size]
    query_embeddings = query_embeddings[gpu_id*query_size:(gpu_id+1)*query_size]
    query_embeddings = torch.tensor(query_embeddings, dtype=torch.double).cuda()

    query_topk_list = [[] for _ in range(query_embeddings.size(0))]

    logging.info(f'GPU {gpu_id} has {query_embeddings.size(0)} queries')

    if len(qid_list) <= 6980:
        passage_batch_size = 128*1024
    else:
        passage_batch_size = torch.cuda.device_count()*512

    global_offset = 0
    for i in range(torch.cuda.device_count()):
        logging.info(f'Loading passage embeddings from GPU {i}')
        passage_embeddings = np.load(os.path.join(output_path, f'passage_embeddings_{i}.npy'))

        total_scores = []
        total_index = []
        for local_offset in tqdm(range(0, passage_embeddings.shape[0], passage_batch_size), desc=f'searching with GPU {gpu_id} on passage block {i}', mininterval=2):
            cur_passage_embeddings = passage_embeddings[local_offset:local_offset+passage_batch_size]
            cur_passage_embeddings = torch.tensor(cur_passage_embeddings, dtype=torch.double).cuda()
            scores = torch.matmul(query_embeddings, cur_passage_embeddings.t())
            scores, index = torch.topk(scores, args.topk, dim=1)
            index = index + (global_offset + local_offset)
            total_scores.append(scores)
            total_index.append(index)

            if len(total_scores) >= 5:
                temp_scores = torch.cat(total_scores, dim=1)
                temp_index = torch.cat(total_index, dim=1)
                _, topk_index = torch.topk(temp_scores, args.topk, dim=1)
                new_scores = torch.gather(temp_scores, dim=1, index=topk_index)
                new_index = torch.gather(temp_index, dim=1, index=topk_index)
                total_scores = [new_scores]
                total_index = [new_index]
                torch.cuda.empty_cache()
        
        total_scores = torch.cat(total_scores, dim=1).cpu()
        total_index = torch.cat(total_index, dim=1).cpu()
        for j in range(total_scores.size(0)):
            cur_scores = total_scores[j].tolist()
            cur_index = total_index[j].tolist()
            query_topk_list[j].extend([(cur_scores[k], cur_index[k]) for k in range(len(cur_scores))])
            query_topk_list[j] = sorted(query_topk_list[j], key=lambda x: x[0], reverse=True)[:args.topk]

        global_offset += passage_embeddings.shape[0]


    results_file = os.path.join(output_path, f'results_{gpu_id}.top{args.topk}')
    with open(results_file, 'w') as f:
        for qid, query_topk in zip(qid_list, query_topk_list):
            for rank, (score, pid) in enumerate(query_topk):
                f.write(f'{qid}\t{pid}\t{rank+1}\n')

def search(args):
    '''Search'''
    logging.info(f'Searching with {torch.cuda.device_count()} GPUs...')
    torch.multiprocessing.spawn(_worker_search, nprocs=torch.cuda.device_count(), args=(args,))
    logging.info(f'Merging results...')
    cache_path = os.path.join(args.output_dir, 'infer_cache')
    results_file = os.path.join(args.output_dir, f'results.top{args.topk}')
    with open(results_file, 'w') as f:
        for i in range(torch.cuda.device_count()):
            cur_results_file = os.path.join(cache_path, f'results_{i}.top{args.topk}')
            with open(cur_results_file, 'r') as cur_f:
                for line in cur_f:
                    f.write(line)
            # os.remove(cur_results_file)

def main():
    '''Main function'''
    # Parse arguments
    args = HfArgumentParser((InferArguments,)).parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(args)
    encode_passages(args)
    encode_queries(args)
    search(args)



if __name__ == '__main__':
    main()