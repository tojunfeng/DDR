from transformers import TrainingArguments
from dataclasses import dataclass, field

@dataclass
class Arguments(TrainingArguments):
    '''Customized TrainingArguments'''

    model_path: str = field(default='model', metadata={'help': 'Path to save model'})
    checkpoint: str = field(default='checkpoint', metadata={'help': 'Path to save checkpoint'})
    message: str = field(default='Hello, world!', metadata={'help': 'Message to print'})

    num_train_epochs: float = field(default=3, metadata={'help': 'Number of training epochs'})
    warmup_steps: int = field(default=1000, metadata={'help': 'Number of warmup steps'})
    per_device_train_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core for training'})
    per_device_eval_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core for evaluation'})
    learning_rate: float = field(default=5e-5, metadata={'help': 'The initial learning rate for AdamW'})
    kd_loss_weight: float = field(default=0.5, metadata={'help': 'Knowledge distillation loss weight'})

    seed: int = field(default=42, metadata={'help': 'Random seed'})

    logging_steps: int = field(default=0.01, metadata={'help': 'Log every n steps'})

    do_data_sample: bool = field(default=False, metadata={'help': 'Sample data'})
    do_kd_loss: bool = field(default=False, metadata={'help': 'Knowledge distillation loss'})
    num_negatives: int = field(default=1, metadata={'help': 'Number of negative passages per query'})
    corpus_file: str = field(default='data/corpus.json', metadata={'help': 'Path to corpus file'})
    train_file: str = field(default='data/train.json', metadata={'help': 'Path to train file'})
    dev_file: str = field(default='data/dev.json', metadata={'help': 'Path to dev file'})
    query_length: int = field(default=32, metadata={'help': 'Max length of query'})
    passage_length: int = field(default=144, metadata={'help': 'Max length of passage'})

    save_strategy: str = field(default='epoch', metadata={'help': 'The checkpoint save strategy'})

    def __post_init__(self):
        self.remove_unused_columns = False
        self.disable_tqdm = True
        self.save_safetensors = False
        self.label_names = ['labels']
        if self.do_data_sample:
            self.num_negatives = 3
            self.per_device_train_batch_size = 32
        return super().__post_init__()


@dataclass
class InferArguments:
    '''Inference arguments'''
    corpus_file: str = field(default='data/corpus.json', metadata={'help': 'Path to corpus file'})
    query_file: str = field(default='data/query.json', metadata={'help': 'Path to query file'})

    topk: int = field(default=10, metadata={'help': 'Top k passages to retrieve'})

    query_length: int = field(default=32, metadata={'help': 'Max length of query'})
    passage_length: int = field(default=144, metadata={'help': 'Max length of passage'})

    do_data_sample: bool = field(default=False, metadata={'help': 'Sample data'})
    output_dir: str = field(default='output', metadata={'help': 'Path to save output'})
    checkpoint: str = field(default='checkpoint', metadata={'help': 'Path to load checkpoint'})
    model_path: str = field(default='model', metadata={'help': 'Path to save model'})
    per_device_eval_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core for evaluation'})


@dataclass
class CrossScoreArguments:
    '''Inference arguments'''
    corpus_file: str = field(default='data/corpus.json', metadata={'help': 'Path to corpus file'})
    train_file: str = field(default='data/query.json', metadata={'help': 'Path to query file'})
    bm25_file: str = field(default='data/bm25.json', metadata={'help': 'Path to bm25 file'})

    query_length: int = field(default=32, metadata={'help': 'Max length of query'})
    passage_length: int = field(default=144, metadata={'help': 'Max length of passage'})
    output_dir: str = field(default='output', metadata={'help': 'Path to save output'})
    model_path: str = field(default='model', metadata={'help': 'Path to save model'})
    per_device_eval_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core for evaluation'})

