import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from critical_error_detection.common.utils import print_stat, format_submission
from critical_error_detection.en_zh.monotransquest_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, \
    monotransquest_config, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FILE = "critical_error_detection/en_zh/data/enzh_majority_train.tsv"
DEV_FILE = "critical_error_detection/en_zh/data/enzh_majority_dev.tsv"
TEST_FILE = "critical_error_detection/en_zh/data/enzh_majority_test_blind.tsv"

train = pd.read_table(TRAIN_FILE, names=['id', 'source', 'translation', 'errors', 'error_labels'], converters={'errors': eval})
dev = pd.read_table(DEV_FILE, names=['id', 'source', 'translation', 'errors', 'error_labels'], converters={'errors': eval})
test = pd.read_table(TEST_FILE, names=['id', 'source', 'translation'])

train = train[['source', 'translation', 'error_labels']]
dev = dev[['source', 'translation', 'error_labels']]
test = test[['id', 'source', 'translation']]

index = test['id'].to_list()
train = train.rename(columns={'source': 'text_a', 'translation': 'text_b', 'error_labels': 'labels'}).dropna()
dev = dev.rename(columns={'source': 'text_a', 'translation': 'text_b', 'error_labels': 'labels'}).dropna()
test = test.rename(columns={'source': 'text_a', 'translation': 'text_b'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

assert (len(index) == 1000)
if monotransquest_config["evaluate_during_training"]:
    if monotransquest_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
        test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))
        for i in range(monotransquest_config["n_fold"]):

            if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(
                    monotransquest_config['output_dir']):
                shutil.rmtree(monotransquest_config['output_dir'])

            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df)
            model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=2,
                                        use_cuda=torch.cuda.is_available(), args=monotransquest_config)
            result, model_outputs, wrong_predictions = model.eval_model(dev)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev_preds[:, i] = model_outputs
            test_preds[:, i] = predictions

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=2, use_cuda=torch.cuda.is_available(),
                                    args=monotransquest_config)
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train_df, eval_df=eval_df)
        model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=2,
                                    use_cuda=torch.cuda.is_available(), args=monotransquest_config)
        result, model_outputs, wrong_predictions = model.eval_model(dev)
        predictions, raw_outputs = model.predict(test_sentence_pairs)

        print("##model_outputs##\n", model_outputs)
        print("##predictions##\n", predictions)
        print("##raw_outputs##\n", raw_outputs)
        print("##result##\n", result)

        dev['predictions'] = np.argmax(model_outputs, axis=1)
        dev['predictions'] = dev['predictions'].replace([0, 1], ['NOT', 'ERR'])
        test['predictions'] = predictions

else:
    model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                args=monotransquest_config)
    model.train_model(train)
    result, model_outputs, wrong_predictions = model.eval_model(dev)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    dev['predictions'] = model_outputs
    dev['predictions'] = dev['predictions'].replace([0, 1], ['NOT', 'ERR'])
    test['predictions'] = predictions

dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
print_stat(dev, 'labels', 'predictions')

format_submission(df=test, index=index, language_pair="en-zh", method="TransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))
