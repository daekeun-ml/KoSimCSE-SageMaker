import json
from pathlib import Path
from typing import List
import os
import torch
import argparse
from classopt import classopt
from more_itertools import chunked
from transformers import AutoTokenizer, logging
from transformers.tokenization_utils import BatchEncoding

from sts import STSEvaluation
from train import SimCSEModel
from datasets import load_dataset
from collections import defaultdict

def parser_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--base_model", type=str, default="klue/roberta-base")
    parser.add_argument("--base_model", type=str, default="BM-K/KoSimCSE-roberta")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    return args

def main(args):

    logging.set_verbosity_error()

    model = SimCSEModel(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    model.eval().to(args.device)

    @torch.inference_mode()
    def encode(texts: List[str]) -> torch.Tensor:
        embs = []
        for text in chunked(texts, args.batch_size):
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            emb = model(**batch.to(args.device))#, use_mlp=False)
            embs.append(emb.cpu())
        return torch.cat(embs, dim=0)

    evaluation = STSEvaluation()
    sts_metrics = evaluation(encode=encode)
    print(sts_metrics)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/sts-metrics.json", "w") as f:
        json.dump(sts_metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parser_args()
    #args.model_path = "./outputs/model.pt"
    main(args)
