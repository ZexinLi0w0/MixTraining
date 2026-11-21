"""Schedule the language-modeling objective alongside GLUE fine-tuning."""

from transformers import Trainer

from data_collator import mask_tokens


class MixTrainer(Trainer):
    def __init__(self, *args, mix_end_epoch=0, mlm_probability=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix_end_epoch = mix_end_epoch
        self.mlm_probability = mlm_probability

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training or (self.state.epoch or 0) >= self.mix_end_epoch:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        inputs = dict(inputs, pretrain_mlm=True)
        if self.model.config.model_type == "bert":
            inputs["mlm_input_ids"], inputs["mlm_gts"] = mask_tokens(
                inputs["input_ids"], self.processing_class, self.mlm_probability
            )
        else:
            inputs["mlm_gts"] = inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, -100)

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.label_smoother is not None:
            # Trainer handles the task labels itself; the model only computed the LM loss.
            loss = loss + outputs.loss
        return (loss, outputs) if return_outputs else loss
