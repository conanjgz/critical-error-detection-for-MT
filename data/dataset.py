import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset

class CEDDataset(Dataset):

    def __init__(self, path, huggingface_model):
        # read from tsv files
        # use the huggingface pretrained tokenizer
        self.df = pd.read_table(path, names=['id', 'source', 'translation', 'errors', 'error_labels'], converters={'errors': eval})
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.pad = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
    def __len__(self):
        # length of the dataset, return the number of examples
        return len(self.df.index)

    def __getitem__(self, idx):
        # get the i-th example in the Dataset
        # use huggingface tokenizer to convert text to Tensors
        # return text Tensors (input_ids, attn_mask, type_ids) and label Tensors
        src_texts = self.df['source']
        trg_texts = self.df['translation']

        src_text = src_texts[idx]
        trg_text = trg_texts[idx]

        # [CLS] sent1 [SEP] sent2 [SEP]

        # encoding = self.tokenizer(src_text, trg_text, return_tensors='pt')
        # e.g. {'input_ids': tensor([[     0,     87,   5161,    398,      2,      2,  13129, 189138,      2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        label = self.df['error_labels'][idx]
        label = 0 if label == 'NOT' else 1
        # label = torch.tensor([int(label)], dtype=torch.long)

        # return {
        #     'input_ids': encoding['input_ids'].squeeze(),
        #     'attention_mask': encoding['attention_mask'].squeeze(),
        #     'labels': label,
        # }
        return (src_text, trg_text, label)