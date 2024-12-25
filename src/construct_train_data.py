import argparse
import jsonlines
import numpy as np
from tqdm import tqdm



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, default='data/processed')
    parser.add_argument('--qrel_file', type=str, default='data/processed')
    parser.add_argument('--output_file', type=str, default='output')
    parser.add_argument('--topk_file', type=str, default='output')
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_qrel(qrel_file):
    qrel = {}
    with open(qrel_file, 'r') as f:
        for l in f:
            qid, _, pid, rel = l.strip().split()
            qid = int(qid)
            pid = int(pid)
            rel = int(rel)
            if qid not in qrel:
                qrel[qid] = []
            qrel[qid].append(pid)
    return qrel

def load_topk(topk_file):
    topk = {}
    with open(topk_file, 'r') as f:
        for l in f:
            qid, pid, rank = l.strip().split()
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            if qid not in topk:
                topk[qid] = [0]*200
            if rank <= 200:
                topk[qid][rank-1] = pid
    return topk

def load_query(query_file):
    query = {}
    with open(query_file, 'r') as f:
        for l in f:
            qid, q = l.strip().split('\t')
            qid = int(qid)
            query[qid] = q
    return query



def main():
    args = config()
    # np.random.seed(args.seed)
    print(args)
    qrel = load_qrel(args.qrel_file)
    topk = load_topk(args.topk_file)
    query = load_query(args.query_file)
    print(f'Loaded {len(qrel)} queries from qrel, {len(topk)} topk lists, {len(query)} queries from query file')
    with jsonlines.open(args.output_file, 'w') as f:
        for qid in tqdm(qrel, desc='constructing training data', mininterval=5):
            if qid not in topk:
                continue
            if qid not in query:
                continue
            positive_list = qrel[qid]
            negative_list = topk[qid]
            negative_list = [pid for pid in negative_list if pid not in positive_list]
            if len(negative_list) < args.depth:
                print(f'Warning: query {qid} has less than {args.depth} negative samples')
                continue
            # np.random.shuffle(negative_list)
            start = 0
            negative_list = negative_list[start:start+args.depth]
            query_str = query[qid]
            f.write({
                'query_id': qid,
                'query': query_str,
                'positives': {
                    'doc_id': positive_list,
                    'score': [1.0 for _ in positive_list]
                },
                'negatives': {
                    'doc_id': negative_list,
                    'score': [0.0 for _ in negative_list]
                }
            })

if __name__ == '__main__':
    main()