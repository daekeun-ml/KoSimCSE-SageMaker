import os
import json
import random
import argparse
import logging
import time, datetime
from typing import Dict, List, Union
from datasets import load_dataset, load_from_disk

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from more_itertools import chunked
from tqdm import tqdm
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer, default_data_collator

from simcse import SimCSEDatasetFromHF, SimCSEModel, set_seed
from sts import STSEvaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[TqdmLoggingHandler()])


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--base_model", type=str, default="klue/roberta-base", help="Model id to use for training.")
    parser.add_argument("--dataset_dir", type=str, default="../dataset-sup-train")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--save_path", type=str, default="../model")

    # add training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training.")
    # the number of epochs is 1 for Unsup-SimCSE, and 3 for Sup-SimCSE in the paper
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate to use for training.")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.05) # see Table D.1 of the paper
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", default=False, action="store_true")
    
    args = parser.parse_known_args()
    return args


def main(args):
    transformers.logging.set_verbosity_error()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = load_from_disk(args.dataset_dir)

    if args.debug:
        train_num_samples = 5000
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_num_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = SimCSEModel(args.base_model).to(args.device)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers and pin_memory are for speeding up training
        num_workers=16,
        pin_memory=True,
        # batch_size varies in the last batch because
        # the last batch size will be the number of remaining samples (i.e. len(train_dataloader) % batch_size)
        # to avoid unstablity of contrastive learning, we drop the last batch
        drop_last=True,
    )
    # FYI: huggingface/transformers' AdamW implementation is deprecated and you should use PyTorch's AdamW instead.
    # see: https://github.com/huggingface/transformers/issues/3407
    #      https://github.com/huggingface/transformers/issues/18757
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # reference implementation uses a linear scheduler with warmup, which is a default scheduler of transformers' Trainer
    # with num_training_steps = 0 (i.e. no warmup)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        # len(train_dataloader) is the number of steps in one epoch
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # evaluation class for STS task
    # we use a simple cosine similarity as a semantic similarity
    # and use Spearman's correlation as an evaluation metric
    # see: `sts.py`
    sts = STSEvaluation()

    # encode sentences (List[str]) and output embeddings (Tensor)
    # this is for evaluation
    @torch.inference_mode()
    def encode(texts: List[str]) -> torch.Tensor:
        embs = []
        model.eval()
        for text in chunked(texts, args.batch_size * 8):
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # SimCSE uses MLP layer only during training
            # in this implementation, we use `model.training` to switch between training and evaluation
            emb = model(**batch.to(args.device))
            embs.append(emb.cpu())
        # shape of output: (len(texts), hidden_size)
        return torch.cat(embs, dim=0)

    # evaluation before training
    model.eval()
    best_sts = sts.dev(encode=encode)["avg"]
    best_step = 0

    # evaluate the model and store metrics before training
    # this is important to check the appropriateness of training procedure
    logging.info(f"epoch: {0:>3};\tstep: {0:>6};\tloss: {' '*9}nan;\tAvg. STS: {best_sts:.5f};")
    logs: List[Dict[str, Union[int, float]]] = [
        {
            "epoch": 0,
            "step": best_step,
            "loss": None,
            "sts": best_sts,
        }
    ]

    # finally, start training!
    for epoch in range(args.num_epochs):
        model.train()
        
        epoch_pbar = tqdm(total=len(train_dataloader), colour="blue", leave=True, desc=f"Training epoch {epoch}")    

        for step, batch in enumerate(train_dataloader):
            
            # transfer batch to the device
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

            z1_input_ids = input_ids[:, 0, :].to(args.device)
            z2_input_ids = input_ids[:, 1, :].to(args.device)
            z3_input_ids = input_ids[:, 2, :].to(args.device)
            z1_attention_mask = attention_mask[:, 0, :].to(args.device)
            z2_attention_mask = attention_mask[:, 1, :].to(args.device)
            z3_attention_mask = attention_mask[:, 2, :].to(args.device)
            z1_token_type_ids = token_type_ids[:, 0, :].to(args.device)
            z2_token_type_ids = token_type_ids[:, 1, :].to(args.device)
            z3_token_type_ids = token_type_ids[:, 2, :].to(args.device)

            # embedding
            z1_emb = model.forward(input_ids=z1_input_ids, attention_mask=z1_attention_mask, token_type_ids=z1_token_type_ids)
            z2_emb = model.forward(input_ids=z2_input_ids, attention_mask=z2_attention_mask, token_type_ids=z2_token_type_ids)
            z3_emb = model.forward(input_ids=z3_input_ids, attention_mask=z3_attention_mask, token_type_ids=z3_token_type_ids)

            # SimCSE training objective:
            #    maximize the similarity between the same sentence
            # => make diagonal elements most similar        
            # FYI: SimCSE is sensitive for the temperature parameter.
            # see Table D.1 of the paper        
            z1_z2_sim = F.cosine_similarity(z1_emb.unsqueeze(1), z2_emb.unsqueeze(0), dim=-1) / args.temperature
            z1_z3_sim = F.cosine_similarity(z1_emb.unsqueeze(1), z3_emb.unsqueeze(0), dim=-1) / args.temperature
            sim_matrix = torch.cat([z1_z2_sim, z1_z3_sim], dim=1)

            # labels := [0, 1, 2, ..., batch_size - 1]
            # labels indicate the index of the diagonal element (i.e. positive examples)
            labels = torch.arange(args.batch_size).long().to(args.device)
            # it may seem strange to use Cross-Entropy Loss here.
            # this is a shorthund of doing SoftMax and maximizing the similarity of diagonal elements
            loss = F.cross_entropy(sim_matrix, labels)

            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_pbar.update(1)
            # for every `args.eval_steps` steps, perform evaluation on STS task and print logs
            if (step + 1) % args.eval_steps == 0 or (step + 1) == len(train_dataloader):
                model.eval()
                # evaluate on the STS-B development set
                sts_score = sts.dev(encode=encode)["avg"]

                # you should use the best model for the evaluation to avoid using overfitted model
                # FYI: https://github.com/princeton-nlp/SimCSE/issues/62
                if best_sts < sts_score:
                    best_sts = sts_score
                    best_step = step + 1
                    # only save the best performing model
                    torch.save(model.state_dict(), f"{args.output_dir}/model.pt")

                logging.info(
                    f"epoch: {epoch:>3};\tstep: {step+1:>6};\tloss: {loss.item():.10f};\tAvg. STS: {sts_score:.5f};"
                )
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss.item(),
                        "sts": sts_score,
                    }
                )
                pd.DataFrame(logs).to_csv(f"{args.output_dir}/logs.csv", index=False)

                # if you want to see the changes of similarity matrix, uncomment the following line
                # tqdm.write(str(sim_matrix))
                model.train()
            
    # save num_epochs, steps, losses, and STS dev scores
    with open(f"{args.output_dir}/dev-metrics.json", "w") as f:
        data = {
            "best-step": best_step,
            "best-sts": best_sts,
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

#     sts_metrics = sts(encode=encode)
#     with open(f"{args.output_dir}/sts-metrics.json", "w") as f:
#         json.dump(sts_metrics, f, indent=2, ensure_ascii=False)

    with open(f"{args.output_dir}/config.json", "w") as f:
        data = {k: v if type(v) in [int, float] else str(v) for k, v in vars(args).items()}
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args, _ = parse_args()
    
    start = time.time()
    main(args)
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs)
    logging.info(f"Elapsed time: {result}")