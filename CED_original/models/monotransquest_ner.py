import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class MonoTransQuestModel(nn.Module):

    # TODO modify input arguments for custom use
    def __init__(self, huggingface_model, tokenizer, **kwargs):
        super().__init__()
        # https://huggingface.co/transformers/model_doc/auto.html?highlight=automodel#transformers.AutoModelForSequenceClassification
        self.config = AutoConfig.from_pretrained(huggingface_model, num_labels=2)
        # print(self.config.num_labels)
        self.model = AutoModel.from_pretrained(huggingface_model, config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size + 4, self.config.hidden_size + 4),
            torch.tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size + 4, self.config.hidden_size + 4),
        )

        # initialization
        

    def forward(self, batch):
        outputs = self.model(**batch)
        pooler_output = outputs[1]
        torch.cat(pooler_output, ner_freq, dim=1)
        logits = self.classifier(pooler_output)
        loss, logits = outputs[0:2]
        return loss, logits
