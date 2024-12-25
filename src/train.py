import logging
import torch
import random
from biencoder_trainer import BiencoderTrainer
from config import Arguments
from data import RetrievalDataset, RetrievalCollator
from model import BiencodeModel
from transformers.hf_argparser import HfArgumentParser
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback
from utils import LoggerCallback
from functools import partial

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _compute_metrics(args, pred):
    '''Compute metrics'''
    scores = torch.Tensor(pred.predictions[0])
    argmax_scores = torch.argmax(scores, dim=1)
    labels = torch.Tensor(pred.predictions[1])
    acc = (argmax_scores == labels).float().mean()
    return {"acc": acc}


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)
    if args.local_rank in [-1, 0]:
        logging.info(f"Running with the following parameters: {args}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    retrieval_dataset = RetrievalDataset(args, tokenizer)
    train_dataset = retrieval_dataset.train_dataset
    dev_dataset = retrieval_dataset.dev_dataset
    data_collator = RetrievalCollator(tokenizer)
    model = BiencodeModel.build(args)
    trainer = BiencoderTrainer(
        args=args,
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics = partial(_compute_metrics, args)
    )
    trainer.pop_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    trainer.train()
    trainer.save_model(args.checkpoint)

if __name__ == '__main__':
    main()