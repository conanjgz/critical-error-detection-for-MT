import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset

class CEDDataset(Dataset):

    def __init__(self, path, huggingface_model):
        # read from tsv files
        if 'blind' in path:
            self.df = pd.read_table(path, names=['id', 'source', 'translation'])
            self.is_inference = True
        else:
            self.df = pd.read_table(path, names=['id', 'source', 'translation', 'errors', 'error_labels'])
            self.is_inference = False
        
    def __len__(self):
        # length of the dataset, return the number of examples
        return len(self.df.index)

    def __getitem__(self, idx):
        # get the i-th example in the Dataset
        src_texts = self.df['source']
        trg_texts = self.df['translation']

        src_text = src_texts[idx]
        trg_text = trg_texts[idx]

        if self.is_inference:
            label = 0
        else:
            label = self.df['error_labels'][idx]
            label = 0 if label == 'NOT' else 1

        return (src_text, trg_text, label)

    def get_label_count(self):
        return list(self.df['error_labels'].value_counts())
    
    def get_label_list(self):
        label_list = self.df['error_labels']
        return [0 if label == 'NOT' else 1 for label in label_list]
