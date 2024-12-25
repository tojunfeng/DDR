import random
import logging
import torch
from dataclasses import dataclass
from typing import Dict, List
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from transformers.file_utils import PaddingStrategy


def group_passage_id(examples, num_negatives):
    '''Group passages by passage_id'''
    positive_ids = []
    negative_ids = []
    for example_positive in examples['positives']:
        temp = [(int(pos_id), float(score)) for pos_id, score in zip(example_positive['doc_id'], example_positive['score'])]
        rand_index = random.randint(0, len(temp) - 1)
        chosen_tuple = temp[rand_index]
        positive_ids.append(chosen_tuple)

    for example_negative in examples['negatives']:
        temp = [(int(neg_id), float(score)) for neg_id, score in  zip(example_negative['doc_id'], example_negative['score'])]
        if len(temp) < num_negatives:
            temp += temp * (num_negatives // len(temp))
        selected_index = random.sample(range(len(temp)), num_negatives)
        temp = [temp[i] for i in selected_index]
        # temp = temp[:num_negatives]
        negative_ids.append(temp)
    
    input_passage_id = []
    input_passage_score = []
    for positive_id, negative_id in zip(positive_ids, negative_ids):
        input_passage_id.append(positive_id[0])
        temp = [positive_id[1]]
        for neg_id, score in negative_id:
            input_passage_id.append(neg_id)
            temp.append(score)
        input_passage_score.append(temp)
    input_passage_score = torch.tensor(input_passage_score, dtype=torch.float32)
    assert len(input_passage_id) == len(positive_ids) * (1 + num_negatives), 'Error in grouping passage_id'
    return input_passage_id, input_passage_score


class RetrievalDataset:

    def __init__(self, args, tokenizer: AutoTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.num_negatives = args.num_negatives
        self.corpus = load_dataset('json', data_files=args.corpus_file)['train']
        self.train_dataset, self.dev_dataset = self.get_train_dev_dataset()

    def get_train_dev_dataset(self):
        train_dataset = load_dataset('json', data_files=self.args.train_file)['train']
        train_dataset.set_transform(self._transform_func)
        if self.args.do_data_sample:
            logging.info('Sampling data')
            train_dataset = train_dataset.select(range(10000))
        for idx in random.sample(range(len(train_dataset)), 3):
            logging.info(f' *** Example {idx} ***')
            logging.info(f"query ids: {train_dataset[idx]['query']['input_ids']}")
            logging.info(f"passage ids: {train_dataset[idx]['passage']['input_ids']}")

        
        dev_dataset = load_dataset('json', data_files=self.args.dev_file)['train']
        dev_dataset.set_transform(self._transform_func)
        return train_dataset, dev_dataset

    def _transform_func(self, examples):
        '''Transform a batch of examples to features'''
        input_passage_id, input_passage_score = group_passage_id(examples, self.num_negatives)
        input_titles = [self.corpus[pid]['title'] for pid in input_passage_id]
        input_texts = [self.corpus[pid]['contents'] for pid in input_passage_id]

        # passages
        batched_passage_encodings = self.tokenizer(
            input_titles,
            text_pair=input_texts,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            max_length=self.args.passage_length
        )

        # queries
        batched_query_encodings = self.tokenizer(
            examples['query'],
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            max_length=self.args.query_length
        )

        return {
            'query': [batched_query_encodings],
            'passage': [batched_passage_encodings],
            'kl_label': [input_passage_score]
        }


@dataclass
class RetrievalCollator(DataCollatorWithPadding):

    def __call__(self, features):
        # print(features)
        assert len(features) == 1, 'One batch at a time!'
        features = features[0]
        # padding
        features['query'] = self.tokenizer.pad(
            features['query'],
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        features['passage'] = self.tokenizer.pad(
            features['passage'],
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        features['labels'] = torch.zeros(features['query'].input_ids.shape[0], dtype=torch.long)
        return features
            







