# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import *

import math
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from utils import contrastive_mmd


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

        self.register_parameter('decoder_weight', self.decoder.weight)
        # self.register_parameter('decoder_bias', self.decoder.bias)  # not necessary

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


# @add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class ModifiedRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `ModifiedRoberta` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.verbalizer = nn.Linear(config.vocab_size+config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, 2)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()
    
    def verbalizer_uniform_init(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if 'verbalizer.weight' in n:
                    y = 1.0 / math.sqrt(p.shape[1])
                    p.data.uniform_(-y, y)
                if 'verbalizer.bias' in n:
                    p.data.fill_(0.)
                
            # mean, std, lower, upper = 0, 0.02, -0.04, 0.04
            # l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            # u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            # p.uniform_(2 * l - 1, 2 * u - 1)
            # p.erfinv_()
            # p.mul_(std * math.sqrt(2.))
            # p.add_(mean)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     mask="<mask>",
    #     expected_output="' Paris'",
    #     expected_loss=0.1,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        trigger_labels: Optional[torch.LongTensor] = None,
        scaling_trigger: Optional[float] = 1.0,
        zero_shot: Optional[bool] = False,
        scaling_contrastive: Optional[float] = 1.0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        sequence_output = outputs[0]
        trigger_logits = self.classifier(sequence_output)

        attention_weights = torch.mean(torch.sum(outputs[-1][-1].detach(), dim=-2), dim=1)
        context_start_pose = (input_ids[0] == 2).max(dim=-1)[-1] + 2
        context_end_poses = (input_ids == 1).max(dim=-1)[-1] - 1
        context_end_poses[context_end_poses==-1] = input_ids.shape[1] - 1
        attention_weights[:, :context_start_pose] = -1e9
        for i in range(input_ids.shape[0]):
            attention_weights[i, context_end_poses[i]:] = -1e9
        # attention_weights = torch.softmax(attention_weights, dim=-1)
        trigger_weights = torch.softmax(trigger_logits, dim=-1)[:, :, 1] * attention_weights
        trigger_probs = torch.softmax(trigger_weights, dim=-1).unsqueeze(-1)
        trigger_rep = (sequence_output * trigger_probs).sum(dim=1)

        mask_pos_ids = (input_ids == 50264).nonzero()[:, 1]  # 50264 is the id for <mask> 
        assert len(input_ids) == len(mask_pos_ids)
        prediction_scores = self.lm_head(sequence_output[torch.arange(len(input_ids)), mask_pos_ids])
        trigger_aware_logits = torch.cat((prediction_scores, trigger_rep), dim=-1)
        class_logits = self.verbalizer(gelu(trigger_aware_logits))

        total_loss = None
        if class_labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(class_logits.view(-1, self.config.num_labels), class_labels.view(-1))
            
            if trigger_labels is not None:
                loss_fct = CrossEntropyLoss()
                total_loss += scaling_trigger * loss_fct(trigger_logits.view(-1, 2), trigger_labels.view(-1))
            
            if zero_shot:
                total_loss += scaling_contrastive * contrastive_mmd(trigger_aware_logits, class_labels)
        
        output = (class_logits, trigger_logits, trigger_aware_logits)
        return ((total_loss,) + output) if total_loss is not None else output