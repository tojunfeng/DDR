import torch
import logging
from config import CrossScoreArguments
from transformers import HfArgumentParser
from cross_encoder_infer import encode_score

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    set_seed(42)
    args = HfArgumentParser((CrossScoreArguments,)).parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    encode_score(args)

