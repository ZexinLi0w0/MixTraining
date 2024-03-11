
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import Optional, Tuple, Union
from transformers.utils import logging
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

logger = logging.get_logger(__name__)


class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pretrain_mlm: Optional[bool] = None, # LM
        mlm_input_ids: Optional[torch.Tensor] = None, # LM
        mlm_attention_mask: Optional[torch.Tensor] = None, # LM
        mlm_gts: Optional[torch.Tensor] = None, # LM
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
                
        # Pre-training MLM
        if pretrain_mlm:
            transformer_outputs = self.transformer(
                mlm_input_ids,
                past_key_values=past_key_values,
                attention_mask=mlm_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)
            # move labels to correct device to enable model parallelism
            mlm_gts = mlm_gts.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = mlm_gts[..., 1:].contiguous()
            # Flatten the tokens
            lm_loss = None
            if mlm_gts is not None: # (B x T)
                loss_fct = CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # print("CLM loss: ", lm_loss)
                loss += lm_loss # add loss of LM to total loss
                
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        
        
        
        
if __name__ == "__main__":
    import numpy as np
    from itertools import chain
    from transformers import AutoTokenizer, AutoConfig
    from data_collator import default_data_collator
    
    # Create a list of texts and a list of labels
    texts = [
        "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.",
        "The sky is blue due to the shorter wavelength of blue light.",
    ]
    labels = [1, 0]
    
    # Load tokenizer & model
    config = AutoConfig.from_pretrained("gpt2")
    config.pad_token_id = config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("gpt2", config=config)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=config).to("cuda")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize texts
    block_size = tokenizer.model_max_length
    max_seq_length = 128
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length,
        return_tensors="pt",
        # return_special_tokens_mask=True,
    )
    total_length = inputs['input_ids'].shape[1]
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    concatenated_examples = {k: list(chain(*inputs[k])) for k in inputs.keys()}
    print("Concatenated examples: ", concatenated_examples.keys())
    lm_inputs = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    print("LM inputs: ", lm_inputs)
    inputs["mlm_gts"] = lm_inputs["input_ids"].copy()
    inputs['mlm_attention_mask'] = lm_inputs['attention_mask'].copy()
    inputs["mlm_input_ids"] = lm_inputs["input_ids"].copy()
    
    print("Initial inputs: ", inputs)
    # Prepare a batch (in this case, it's just your inputs)
    batch = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
    
    # CLM
    data_collator = default_data_collator
    mlm_batch = data_collator(batch)
    inputs['labels'] = torch.tensor(labels)
    inputs['mlm_gts'] = mlm_batch['mlm_gts']
    inputs["mlm_attention_mask"] = mlm_batch['mlm_attention_mask']
    inputs["mlm_input_ids"] = mlm_batch['mlm_input_ids']
    print("Final inputs: ", inputs)
    inputs = inputs.to("cuda")
    
    # Forward pass
    print("Forward pass (fine-tuning only)")
    outputs = model(**inputs)
    print("loss: ", outputs.loss)
    
    print("Forward pass (with pre-training)")
    outputs = model(**inputs, pretrain_mlm=True)
    print("loss: ", outputs.loss)
    