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
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size + 4, self.config.num_labels),
        )
        

    def forward(self, batch):
        outputs = self.model(**{k:batch[k] for k in ('input_ids', 'attention_mask')})

        pooler_output = outputs[1]
        print(batch['ner'])
        cat_output = torch.cat((pooler_output, batch['ner']), dim=1)
        print(cat_output)
        logits = self.classifier(cat_output)

        loss_fct = nn.CrossEntropyLoss(weight=None)
        loss = loss_fct(logits.view(-1, self.config.num_labels), batch['labels'].view(-1))
        
        return loss, logits
