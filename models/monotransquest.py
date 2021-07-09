import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, AutoConfig

class MonoTransQuestModel(nn.Module):

    # TODO modify input arguments for custom use
    def __init__(self, huggingface_model="xlm-roberta-base", **kwargs):
        super().__init__()
        # https://huggingface.co/transformers/model_doc/auto.html?highlight=automodel#transformers.AutoModelForSequenceClassification
        self.config = AutoConfig.from_pretrained(huggingface_model, num_labels=2)
        print(self.config.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(huggingface_model, config=self.config)

        # initialization
        

    def forward(self, batch):
        outputs = self.model(**batch)
        loss, logits = outputs[0:2]
        return loss, logits
        