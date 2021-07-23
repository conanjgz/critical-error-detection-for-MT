import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class MonoTransQuestModel(nn.Module):
    def __init__(self, huggingface_model, tokenizer, dropout=0.1, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(huggingface_model,
                                                 num_labels=2,
                                                 attention_probs_dropout_prob=dropout,
                                                 hidden_dropout_prob=dropout)
        self.model = AutoModel.from_pretrained(huggingface_model, config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size + 8, self.config.hidden_size + 8), # 8 here is the length of the vector added to hidden states
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size + 8, self.config.num_labels), # 8 here is the length of the vector added to hidden states
        )
        self.initialise()
        

    def initialise(self):
        for param in self.classifier.parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.xavier_uniform(param)


    def forward(self, batch):
        outputs = self.model(**{k:batch[k] for k in ('input_ids', 'attention_mask')})

        pooler_output = outputs[1]
        cat_output = torch.cat((pooler_output, batch['ner']), dim=1)
        logits = self.classifier(cat_output)

        loss_fct = nn.CrossEntropyLoss(weight=None)
        loss = loss_fct(logits.view(-1, self.config.num_labels), batch['labels'].view(-1))
        
        return loss, logits
