
import json
import os
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SimCSEDatasetFromText(Dataset):
    path: Path
    data: List[str] = None

    # For simplicity, this dataset class is designed to tokenize text for each loop,
    # but if performance is more important, you should tokenize all text in advance.
    def __post_init__(self):
        self.data = []
        with self.path.open() as f:
            # to prevent whole text into memory at once
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
    
    
class SimCSEDatasetFromHF(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]["sentence"]

    def __len__(self) -> int:
        return len(self.data)
    

class SimCSEModel(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        # you can use any models
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(base_model)

        # define additional MLP layer
        # see Section 6.3 of the paper for more details
        # refenrece: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/simcse/models.py#L19
        self.hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        # RoBERTa variants don't have token_type_ids, so this argument is optional
        token_type_ids: Tensor = None,
    ) -> Tensor:
        # shape of input_ids: (batch_size, seq_len)
        # shape of attention_mask: (batch_size, seq_len)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # take representations of [CLS] token
        # we only implement the best performing pooling, [CLS], for simplicity
        # you can easily extend to other poolings (such as mean pooling or max pooling) by edting this line
        # shape of last_hidden_state: (batch_size, seq_len, hidden_size)
        emb = outputs.last_hidden_state[:, 0]

        # original SimCSE uses MLP layer only during training
        # see: Table 6 of the paper
        # this trick is a bit complicated, so you may omit it when training your own model
        if self.training:
            emb = self.dense(emb)
            emb = self.activation(emb)
        # shape of emb: (batch_size, hidden_size)
        return emb