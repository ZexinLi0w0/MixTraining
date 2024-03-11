
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import Optional, Tuple, Union
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead


class BertForSequenceClassification(BertPreTrainedModel):
    """
    Bert Model transformer with a sequence classification/regression head on top 
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModel(config)  # Load pretrained bert
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Pre-trained layer
        self.mlm_head = BertOnlyMLMHead(config)
        self.nsp_head = BertOnlyNSPHead(config)
        # Fine-tuning layer
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pretrain_mlm: Optional[bool] = None, # MLM
        mlm_input_ids: Optional[torch.Tensor] = None, # MLM
        mlm_gts: Optional[torch.Tensor] = None, # MLM
        # pretrain_nsp: Optional[bool] = None, # NSP
        # nsp_input_ids: Optional[torch.Tensor] = None, # NSP
        # nsp_token_type_ids: Optional[torch.Tensor] = None, # NSP
        # nsp_attention_mask: Optional[torch.Tensor] = None, # NSP
        # nsp_labels: Optional[torch.Tensor] = None, # NSP
        **kwargs, # MLM & NSP
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            # print("CLS loss: ", loss)
        
        # Pre-training MLM
        if pretrain_mlm:
            outputs = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            prediction_scores = self.mlm_head(outputs[0]) # (B x T x V)
            masked_lm_loss = None
            if mlm_gts is not None: # (B x T)
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_gts.view(-1))
                # print("MLM loss: ", masked_lm_loss)
                loss += masked_lm_loss # add loss of MLM to total loss

        # # Pre-training NSP
        # if pretrain_nsp:
        #     outputs = self.bert(
        #         nsp_input_ids,
        #         attention_mask=nsp_attention_mask,
        #         token_type_ids=nsp_token_type_ids,
        #         # position_ids=position_ids,
        #         # head_mask=head_mask,
        #         # inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #     pooled_output = outputs[1]
        #     if "next_sentence_label" in kwargs:
        #         warnings.warn(
        #             "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
        #             " `labels` instead.",
        #             FutureWarning,
        #         )
        #         nsp_labels = kwargs.pop("next_sentence_label")
        #     seq_relationship_scores = self.nsp_head(pooled_output)
        #     next_sentence_loss = None
        #     if nsp_labels is not None:
        #         loss_fct = CrossEntropyLoss()
        #         next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), nsp_labels.view(-1))
        #         # print("NSP loss: ", next_sentence_loss)
        #         loss += next_sentence_loss # add loss of NSP to total loss 
                
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
        
        
        
if __name__ == "__main__":
    import numpy as np
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    
    # Create a list of texts and a list of labels
    texts = [
        "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.",
        "The sky is blue due to the shorter wavelength of blue light.",
    ]
    labels = [1, 0]
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to("cuda")
    
    # Tokenize texts
    max_seq_length = 128
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    # print("Initial inputs: ", inputs)
    # Prepare a batch (in this case, it's just your inputs)
    batch = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
    
    # MLM
    mlm_probability = 0.15
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=None,
    )
    mlm_batch = data_collator(batch)
    
    inputs['labels'] = torch.tensor(labels)
    inputs['mlm_gts'] = mlm_batch['labels']
    inputs["mlm_input_ids"] = mlm_batch['input_ids']
    
    # # NSP
    # nsp_inputs = tokenizer(texts[0], texts[1], return_tensors="pt")
    # nsp_inputs["labels"] = torch.tensor([1])
    # # print("NSP inputs: ", nsp_inputs)
    # inputs['nsp_input_ids'] = nsp_inputs.input_ids
    # inputs['nsp_token_type_ids'] = nsp_inputs.token_type_ids
    # inputs['nsp_attention_mask'] = nsp_inputs.attention_mask
    # inputs['nsp_labels'] = nsp_inputs.labels
    # print("Final inputs: ", inputs)
    inputs = inputs.to("cuda")
    
    # Forward pass
    print("Forward pass (fine-tuning only)")
    outputs = model(**inputs)
    print("loss: ", outputs.loss)
    
    print("Forward pass (with pre-training)")
    outputs = model(**inputs, pretrain_mlm=True)
    print("loss: ", outputs.loss)
    