from transformers import XLMRobertaTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from data import *
from models import *
from utils import *


def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        pred_labels = []
        train_loss = 0
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            loss, logits = model(batch)
            train_loss += loss.item()
            pred_labels += torch.argmax(torch.sigmoid(logits), dim=1).tolist()

    return pred_labels


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0
    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]
    n_param_all = n_param_learnable + n_param_frozen
    print("# parameters: {} ({} learnable)".format(n_param_all, n_param_learnable))
    return n_param_all


def format_submission(df, language_pair, method, index, path, model_size, num_of_params, index_type=None):

    if index_type is None:
        index = index

    elif index_type == "Auto":
        index = range(0, df.shape[0])

    predictions = df['predictions']
    with open(path, 'w') as f:
        f.write("%s\n" % str(model_size))
        f.write("%s\n" % str(num_of_params))
        for number, prediction in zip(index, predictions):
            text = language_pair + "\t" + method + "\t" + str(number) + "\t" + str(prediction)
            f.write("%s\n" % text)


def generate_submission_file(testset_path, best_model_dir):
    tokenizer = XLMRobertaTokenizer.from_pretrained(best_model_dir)
    df = pd.read_table(testset_path, names=['id', 'source', 'translation'])
    index = df['id'].to_list()

    test_dataset = CEDDataset(testset_path, "xlm-roberta-base")
    collate_fn = PadCollate(tokenizer, 100) # here, 100 is max_seq_length
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    model = torch.load(best_model_dir + 'best_model.pt')

    predictions = predict(model, test_loader)
    predictions = [('ERR' if item == 1 else 'NOT') for item in predictions]
    df['predictions'] = predictions

    model_size = os.stat(best_model_dir + 'best_model.pt').st_size
    num_of_params = get_n_params(model)

    if 'enzh' in testset_path:
        format_submission(df=df, index=index, language_pair="en-zh",
                          method="TransQuest", path=best_model_dir + 'predictions.txt',
                          model_size=model_size, num_of_params=num_of_params)
    elif 'enja' in testset_path:
        format_submission(df=df, index=index, language_pair="en-ja",
                          method="TransQuest", path=best_model_dir + 'predictions.txt',
                          model_size=model_size, num_of_params=num_of_params)
    elif 'ende' in testset_path:
        format_submission(df=df, index=index, language_pair="en-de",
                          method="TransQuest", path=best_model_dir + 'predictions.txt',
                          model_size=model_size, num_of_params=num_of_params)
    elif 'encs' in testset_path:
        format_submission(df=df, index=index, language_pair="en-cs",
                          method="TransQuest", path=best_model_dir + 'predictions.txt',
                          model_size = model_size, num_of_params = num_of_params)
    
    return best_model_dir + 'predictions.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="tsv data filename")
    parser.add_argument("-d", help="best model dir")
    args = parser.parse_args()

    CED_TEST_SET = args.f
    BEST_MODEL_DIR = args.d

    submission_file_path = generate_submission_file(CED_TEST_SET, BEST_MODEL_DIR)
    print("Submission file has been saved to: " + submission_file_path)
