import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset

class CEDDataset(Dataset):

    def __init__(self, path, huggingface_model):
        # read from tsv files
        self.df = pd.read_table(path,
                                names=['id', 'source', 'translation', 'errors', 'error_labels', 'NER_src', 'NER_trg'],
                                converters={'NER_src': eval, 'NER_trg': eval})
        
    def __len__(self):
        # length of the dataset, return the number of examples
        return len(self.df.index)

    def __getitem__(self, idx):
        # get the i-th example in the Dataset
        src_texts = self.df['source']
        trg_texts = self.df['translation']

        src_text = src_texts[idx]
        trg_text = trg_texts[idx]

        src_ner_dict = self.df.at[idx, 'NER_src']
        trg_ner_dict = self.df.at[idx, 'NER_trg']

        count_dat_src = 0
        count_nrp_src = 0
        count_dat_trg = 0
        count_nrp_trg = 0

        for ner_token in src_ner_dict.values():
            if ner_token[2] == 'DATE':
                count_dat_src += 1
            elif ner_token[2] == 'NORP' or ner_token[2] == 'GPE':
                count_nrp_src += 1
        
        for ner_token in src_ner_dict.values():
            if ner_token[2] == 'DATE':
                count_dat_trg += 1
            elif ner_token[2] == 'NORP' or ner_token[2] == 'GPE':
                count_nrp_trg += 1

        ner = [count_dat_src, count_nrp_src, count_dat_trg, count_nrp_trg]


        label = self.df['error_labels'][idx]
        label = 0 if label == 'NOT' else 1

        return (src_text, trg_text, ner, label)
