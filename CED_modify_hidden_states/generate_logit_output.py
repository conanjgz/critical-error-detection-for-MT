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

    return logits.tolist()


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


def generate_submission_file(testset_path, best_model_dir, output_filename):
    tokenizer = XLMRobertaTokenizer.from_pretrained(best_model_dir)
    df = pd.read_table(testset_path, usecols=[0, 1, 2], names=['id', 'source', 'translation'])
    index = df['id'].to_list()

    test_dataset = CEDDataset(testset_path, "xlm-roberta-base")
    collate_fn = PadCollate(tokenizer, 100) # here, 100 is max_seq_length
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn)
    
    model = torch.load(best_model_dir + 'best_model.pt')

    predictions = predict(model, test_loader)
    df['predictions'] = predictions

    model_size = os.stat(best_model_dir + 'best_model.pt').st_size
    num_of_params = get_n_params(model)

    if 'enzh' in testset_path:
        if not os.path.exists('output/enzh/'):
            os.makedirs('output/enzh/')
        format_submission(df=df, index=index, language_pair="en-zh",
                          method="TransQuest", path='output/enzh/' + output_filename,
                          model_size=model_size, num_of_params=num_of_params)
        return 'output/enzh/' + output_filename

    elif 'enja' in testset_path:
        if not os.path.exists('output/enja/'):
            os.makedirs('output/enja/')
        format_submission(df=df, index=index, language_pair="en-ja",
                          method="TransQuest", path='output/enja/' + output_filename,
                          model_size=model_size, num_of_params=num_of_params)
        return 'output/enja/' + output_filename

    elif 'ende' in testset_path:
        if not os.path.exists('output/ende/'):
            os.makedirs('output/ende/')
        format_submission(df=df, index=index, language_pair="en-de",
                          method="TransQuest", path='output/ende/' + output_filename,
                          model_size=model_size, num_of_params=num_of_params)
        return 'output/ende/' + output_filename

    elif 'encs' in testset_path:
        if not os.path.exists('output/encs/'):
            os.makedirs('output/encs/')
        format_submission(df=df, index=index, language_pair="en-cs",
                          method="TransQuest", path='output/encs/' + output_filename,
                          model_size = model_size, num_of_params = num_of_params)
        return 'output/encs/' + output_filename


if __name__ == '__main__':
    """

    How to run:
    E.g.
    >> python generate_logit_output.py -f data_with_NER_feature/enzh_majority_test_blind_ner.tsv -d model_2021_07_26_11_46_10_enzh/ -n filename.txt

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="tsv data file path")
    parser.add_argument("-d", help="best model dir")
    parser.add_argument("-n", help="logits output filename, end with .txt")
    args = parser.parse_args()

    CED_TEST_SET = 'wmt21_official_data/' + args.f
    BEST_MODEL_DIR = 'output/temp/' + args.d
    FILENAME = args.n

    submission_file_path = generate_submission_file(CED_TEST_SET, BEST_MODEL_DIR, FILENAME)
    print("Submission file has been saved to: " + submission_file_path)
