from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.utils import ModelOutput

from Gazesup_bert_model import SP_Encoder


@dataclass
class GazesupMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scanpath_mlm_loss: Optional[torch.FloatTensor] = None
    scanpath_mlm_logits: Optional[torch.FloatTensor] = None
    bert_last_hidden_state: Optional[torch.FloatTensor] = None
    gaze_token_pos: Optional[torch.LongTensor] = None
    sp_len: Optional[torch.LongTensor] = None
    scanpath_selected_hidden_states: Optional[torch.FloatTensor] = None
    gru_output: Optional[torch.FloatTensor] = None
    scanpath_labels_expanded: Optional[torch.LongTensor] = None


class Gazesup_BERTForMaskedLM(BertPreTrainedModel):
    """
    Minimal MLM-compatible measured-scanpath model.

    Design used here:
    1. Text enters BERT in its original token order.
    2. Measured word-level scanpaths are expanded to token/subtoken positions.
    3. The expanded scanpath selects a token-level sequence that goes through the GRU.
    4. GRU outputs stay in scanpath order and length S.
    5. A simple linear head maps the GRU sequence directly to vocabulary logits.
    6. Original MLM labels with shape (B, T) are expanded to scanpath labels with
       shape (B, S) by indexing them with gaze_token_pos.

    This class does not implement full training pipelines. It only provides the
    forward building block needed for the future MLM-style auxiliary branch.
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.sp_encoder = SP_Encoder(config, scanpath_source="measured")
        self.scanpath_mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def _compute_gaze_token_pos(self, lm_word_ids, measured_word_ids, measured_sp_len):
        sn_len = self.sp_encoder._infer_sn_len_from_word_ids(lm_word_ids).to(lm_word_ids.device)
        gaze_word_pos = self.sp_encoder._prepare_measured_word_scanpath(
            measured_word_ids=measured_word_ids,
            measured_sp_len=measured_sp_len,
            sn_len=sn_len,
        )
        gaze_token_pos, sp_len = self.sp_encoder.convert_word_pos_seq_to_token_pos_seq(
            word_pos=gaze_word_pos.unsqueeze(1),
            sn_len=sn_len.unsqueeze(1),
            word_ids_sn=lm_word_ids,
        )
        return gaze_token_pos.to(torch.long), sp_len.to(torch.long)

    def _select_scanpath_hidden_states(self, bert_last_hidden_state, gaze_token_pos):
        token_ids = torch.arange(
            bert_last_hidden_state.shape[1],
            device=bert_last_hidden_state.device,
        )[None, None, :].expand(gaze_token_pos.shape[0], gaze_token_pos.shape[1], -1)
        one_hot = token_ids - gaze_token_pos.unsqueeze(-1)
        one_hot[one_hot != 0] = 1
        one_hot = 1 - one_hot
        selected = torch.einsum("bij,bki->bkj", bert_last_hidden_state, one_hot.float())
        return self.sp_encoder.dropout(selected)

    def _run_scanpath_gru(self, scanpath_selected_hidden_states, sp_len, bert_last_hidden_state):
        packed = pack_padded_sequence(
            scanpath_selected_hidden_states,
            sp_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        gru_packed_output, _ = self.sp_encoder.gru(
            packed,
            bert_last_hidden_state[:, 0, :].unsqueeze(0).contiguous(),
        )
        gru_output, _ = pad_packed_sequence(
            gru_packed_output,
            batch_first=True,
            total_length=scanpath_selected_hidden_states.shape[1],
        )
        return gru_output

    def _expand_labels_to_scanpath(self, labels, gaze_token_pos, sp_len):
        if labels is None:
            return None

        if labels.dim() != 2:
            raise ValueError(
                "labels for Gazesup_BERTForMaskedLM must have shape (batch_size, seq_len). "
                f"Received shape={tuple(labels.shape)}"
            )

        expanded_labels = labels.gather(1, gaze_token_pos.long())
        scanpath_index = torch.arange(gaze_token_pos.shape[1], device=gaze_token_pos.device)[None, :]
        valid_step_mask = scanpath_index < sp_len.unsqueeze(1)
        expanded_labels = expanded_labels.masked_fill(~valid_step_mask, -100)
        return expanded_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        LM_word_ids=None,
        measured_word_ids=None,
        measured_sp_len=None,
        labels=None,
        return_dict=True,
    ):
        if LM_word_ids is None:
            raise ValueError("LM_word_ids is required for Gazesup_BERTForMaskedLM.")
        if measured_word_ids is None or measured_sp_len is None:
            raise ValueError("measured_word_ids and measured_sp_len are required for Gazesup_BERTForMaskedLM.")

        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_last_hidden_state = bert_outputs.last_hidden_state

        gaze_token_pos, sp_len = self._compute_gaze_token_pos(
            lm_word_ids=LM_word_ids,
            measured_word_ids=measured_word_ids,
            measured_sp_len=measured_sp_len,
        )
        scanpath_selected_hidden_states = self._select_scanpath_hidden_states(
            bert_last_hidden_state=bert_last_hidden_state,
            gaze_token_pos=gaze_token_pos,
        )
        gru_output = self._run_scanpath_gru(
            scanpath_selected_hidden_states=scanpath_selected_hidden_states,
            sp_len=sp_len,
            bert_last_hidden_state=bert_last_hidden_state,
        )
        scanpath_mlm_logits = self.scanpath_mlm_head(gru_output)

        scanpath_mlm_loss = None
        scanpath_labels_expanded = None
        if labels is not None:
            scanpath_labels_expanded = self._expand_labels_to_scanpath(
                labels=labels,
                gaze_token_pos=gaze_token_pos,
                sp_len=sp_len,
            )
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            scanpath_mlm_loss = loss_fct(
                scanpath_mlm_logits.reshape(-1, self.config.vocab_size),
                scanpath_labels_expanded.reshape(-1),
            )

        if not return_dict:
            return (
                scanpath_mlm_loss,
                scanpath_mlm_logits,
                bert_last_hidden_state,
                gaze_token_pos,
                sp_len,
                scanpath_selected_hidden_states,
                gru_output,
                scanpath_labels_expanded,
            )

        return GazesupMaskedLMOutput(
            loss=scanpath_mlm_loss,
            scanpath_mlm_loss=scanpath_mlm_loss,
            scanpath_mlm_logits=scanpath_mlm_logits,
            bert_last_hidden_state=bert_last_hidden_state,
            gaze_token_pos=gaze_token_pos,
            sp_len=sp_len,
            scanpath_selected_hidden_states=scanpath_selected_hidden_states,
            gru_output=gru_output,
            scanpath_labels_expanded=scanpath_labels_expanded,
        )
