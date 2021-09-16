import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset
from random import randint

class CEDDataset(Dataset):

    def __init__(self, path, huggingface_model):
        # read from tsv files
        self.ner = False
        self.tox = False
        if 'blind' in path:
            self.is_inference = True
            if 'ner' in path:
                self.df = pd.read_table(path,
                                        names=['id', 'source', 'translation', 'NER_src', 'NER_trg'],
                                        converters={'NER_src': eval, 'NER_trg': eval})
                self.ner = True
            elif 'tox' in path:
                if 'ende' in path:
                    self.ende = True
                    self.df = pd.read_table(path,
                                            names=['id', 'source', 'translation', 'toxicity_src', 'toxicity_trg'],
                                            converters={'toxicity_src': eval, 'toxicity_trg': eval})
                else:
                    self.ende = False
                    self.df = pd.read_table(path,
                                            names=['id', 'source', 'translation', 'toxicity_src'],
                                            converters={'toxicity_src': eval})
                self.tox = True
            else:
                self.df = pd.read_table(path, names=['id', 'source', 'translation'])
        else:
            self.is_inference = False
            if 'ner' in path:
                self.df = pd.read_table(path,
                                        names=['id', 'source', 'translation', 'errors', 'error_labels', 'NER_src', 'NER_trg'],
                                        converters={'NER_src': eval, 'NER_trg': eval})
                self.ner = True
            elif 'tox' in path:
                if 'ende' in path:
                    self.ende = True
                    self.df = pd.read_table(path,
                                            names=['id', 'source', 'translation', 'errors', 'error_labels', 'toxicity_src', 'toxicity_trg'],
                                            converters={'toxicity_src': eval, 'toxicity_trg': eval})
                else:
                    self.ende = False
                    self.df = pd.read_table(path,
                                            names=['id', 'source', 'translation', 'errors', 'error_labels', 'toxicity_src'],
                                            converters={'toxicity_src': eval})
                self.tox = True
            else:
                self.df = pd.read_table(path, names=['id', 'source', 'translation', 'errors', 'error_labels'])

        
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

        if self.ner:
            src_ner_dict = self.df.at[idx, 'NER_src']
            trg_ner_dict = self.df.at[idx, 'NER_trg']

            ##################################################################
            #---------------------------- 7 types ---------------------------#
            
            # count_org_src = 0
            # count_per_src = 0
            # count_dat_src = 0
            # count_crd_src = 0
            # count_ord_src = 0
            # count_nrp_src = 0
            # count_gpe_src = 0
            # count_org_trg = 0
            # count_per_trg = 0
            # count_dat_trg = 0
            # count_crd_trg = 0
            # count_ord_trg = 0
            # count_nrp_trg = 0
            # count_gpe_trg = 0
            
            # for ner_token in src_ner_dict.values():
            #     if ner_token[2] == 'ORG':
            #         count_org_src += 1
            #     elif ner_token[2] == 'PERSON':
            #         count_per_src += 1
            #     elif ner_token[2] == 'DATE':
            #         count_dat_src += 1
            #     elif ner_token[2] == 'CARDINAL':
            #         count_crd_src += 1
            #     elif ner_token[2] == 'ORDINAL':
            #         count_ord_src += 1
            #     elif ner_token[2] == 'NORP':
            #         count_nrp_src += 1
            #     elif ner_token[2] == 'GPE':
            #         count_gpe_src += 1

            # for ner_token in trg_ner_dict.values():
            #     if ner_token[2] == 'ORG':
            #         count_org_trg += 1
            #     elif ner_token[2] == 'PERSON':
            #         count_per_trg += 1
            #     elif ner_token[2] == 'DATE':
            #         count_dat_trg += 1
            #     elif ner_token[2] == 'CARDINAL':
            #         count_crd_trg += 1
            #     elif ner_token[2] == 'ORDINAL':
            #         count_ord_trg += 1
            #     elif ner_token[2] == 'NORP':
            #         count_nrp_trg += 1
            #     elif ner_token[2] == 'GPE':
            #         count_gpe_trg += 1
            
            # ner = [count_org_src, count_per_src, count_dat_src, count_crd_src, count_ord_src, count_nrp_src, count_gpe_src,
            #        count_org_trg, count_per_trg, count_dat_trg, count_crd_trg, count_ord_trg, count_nrp_trg, count_gpe_trg]

            #----------------------------------------------------------------#
            ##################################################################

            ##################################################################
            #--------------------- aggregrate to 4 types --------------------#

            count_nam_src = 0
            count_dat_src = 0
            count_num_src = 0
            count_nrp_src = 0
            count_nam_trg = 0
            count_dat_trg = 0
            count_num_trg = 0
            count_nrp_trg = 0
            
            for ner_token in src_ner_dict.values():
                if ner_token[2] == 'ORG' or ner_token[2] == 'PERSON':
                    count_nam_src += 1
                elif ner_token[2] == 'DATE':
                    count_dat_src += 1
                elif ner_token[2] == 'CARDINAL' or ner_token[2] == 'ORDINAL':
                    count_num_src += 1
                elif ner_token[2] == 'NORP' or ner_token[2] == 'GPE':
                    count_nrp_src += 1

            for ner_token in trg_ner_dict.values():
                if ner_token[2] == 'ORG' or ner_token[2] == 'PERSON':
                    count_nam_trg += 1
                elif ner_token[2] == 'DATE':
                    count_dat_trg += 1
                elif ner_token[2] == 'CARDINAL' or ner_token[2] == 'ORDINAL':
                    count_num_trg += 1
                elif ner_token[2] == 'NORP' or ner_token[2] == 'GPE':
                    count_nrp_trg += 1

            ner = [count_nam_src, count_dat_src, count_num_src, count_nrp_src,
                   count_nam_trg, count_dat_trg, count_num_trg, count_nrp_trg]

            #----------------------------------------------------------------#
            ##################################################################

            return (src_text, trg_text, ner, label)
        
        elif self.tox:
            src_tox_dict = self.df.at[idx, 'toxicity_src']
            src_tox_dict = dict(sorted(src_tox_dict.items()))
            
            if self.ende:
                trg_tox_dict = self.df.at[idx, 'toxicity_trg']
                trg_tox_dict = dict(sorted(trg_tox_dict.items()))

                tox = [src_tox_dict['TOXICITY'], src_tox_dict['SEVERE_TOXICITY'], trg_tox_dict['TOXICITY'], trg_tox_dict['SEVERE_TOXICITY']]

                return (src_text, trg_text, tox, label)
            else:
                tox = [src_tox_dict['TOXICITY'], src_tox_dict['SEVERE_TOXICITY']]
                return (src_text, trg_text, tox, label)
        else:
            return(src_text, trg_text, label)

    def get_label_count(self):
        return list(self.df['error_labels'].value_counts())

    def get_label_list(self):
        label_list = self.df['error_labels']
        return [0 if label == 'NOT' else 1 for label in label_list]
