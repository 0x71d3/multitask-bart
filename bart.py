import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import (
    BartClassificationHead,
    BartForConditionalGeneration
)


class BartForMultitaskLearning(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)

        self.num_emotions = 6
        self.num_sentiments = 2

        self.emotion_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_emotions,
            config.classif_dropout
        )
        self.model._init_weights(self.emotion_head.dense)
        self.model._init_weights(self.emotion_head.out_proj)

        self.sentiment_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_sentiments,
            config.classif_dropout
        )
        self.model._init_weights(self.sentiment_head.dense)
        self.model._init_weights(self.sentiment_head.out_proj)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=False,
        **unused
    ):
        task = unused.pop("task")

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache
        )

        assert task in ["response", "emotion", "sentiment"]

        if task == "response":
            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
            if lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # TODO(SS): do we need to ignore pad tokens in lm_labels?
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

        else:
            if task == "emotion":
                classification_head = self.emotion_head
                num_labels = self.num_emotions
            else:
                classification_head = self.sentiment_head
                num_labels = self.num_sentiments

            x = outputs[0]  # last hidden state
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
            logits = classification_head(sentence_representation)
            # Prepend logits
            outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
            if lm_labels is not None:  # prepend loss to output,
                loss = F.cross_entropy(logits.view(-1, num_labels), lm_labels.view(-1))
                outputs = (loss,) + outputs

        return outputs
