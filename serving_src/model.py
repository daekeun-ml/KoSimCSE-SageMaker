import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers import AutoModel, AutoTokenizer, logging

class SimCSEConfig(PretrainedConfig):
    def __init__(self, version=1.0, **kwargs):
        self.version = version
        super().__init__(**kwargs)

class SimCSEModel(PreTrainedModel):

    config_class = SimCSEConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(config.base_model)
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