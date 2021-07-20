import logging
import math
import os
import warnings
from datetime import datetime
import configargparse as cfargparse
import torch
import torch.nn as nn
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import *
from models import *
from utils import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG_STEP = 300


def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        pred_labels = []
        gold_labels = []
        train_loss = 0
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            loss, logits = model(batch)
            # print(loss)
            train_loss += loss.item()
            pred_labels += torch.argmax(torch.sigmoid(logits), dim=1).tolist()
            gold_labels += batch['labels'].squeeze().tolist()

        cm, acc, prec, rec, macro_f1 = get_performance(gold_labels, pred_labels)

    return train_loss / len(dataloader), cm, acc, prec, rec, macro_f1


def train_epoch(model, dataloader, optimizer, scheduler):
    logger = logging.getLogger("train_log")

    model.train()
    train_loss = 0
    train_acc = 0
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # training loop over all batches
        optimizer.zero_grad()

        loss, logits = model(batch)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(logits)
        pred_labels = torch.argmax(torch.sigmoid(logits), dim=1)
        gold_labels = batch['labels'].squeeze()
        # print("gold", gold_labels.cpu())
        # print("pred", pred_labels.cpu())
        cm, acc, prec, rec, macro_f1 = get_performance(
            gold_labels.cpu(), pred_labels.cpu())
        train_loss += loss.item()
        train_acc += acc

        if idx % LOG_STEP == 0:
            # PS: this is batch stats (can be biased when batch size is very small)
            logger.info(
                "Train Loss: {:.3f}, tp: {}, fn: {}, fp: {}, tn: {}, Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}, F1: {:.3f}".format(
                    loss, cm[0][0], cm[0][1], cm[1][0], cm[1][1], acc, prec, rec, macro_f1)
            )

    # return averaged training stats
    return train_loss / len(dataloader), train_acc / len(dataloader)

def train(args):
    logger = logging.getLogger("train_log")

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_model)
    print(type(args.ner), args.ner)
    # if args.ner == True:
    #     logger.info("NER tokens applied!")
    #     special_tokens_dict = {'additional_special_tokens': [
    #         '<org>', '</org>', '<per>', '</per>', '<dat>', '</dat>', '<crd>', '</crd>', '<ord>', '</ord>', '<nrp>', '</nrp>', '<gpe>', '</gpe>', '<oth>', '</oth>']}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    # if args.tox == True:
    #     logger.info("TOX tokens applied!")
    #     special_tokens_dict = {'additional_special_tokens': ['<tox>']}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    

    fix_seed(1000)
    # load data
    train_dataset = CEDDataset(args.train_data, args.huggingface_model)
    valid_dataset = CEDDataset(args.valid_data, args.huggingface_model)
    collate_fn = PadCollate(tokenizer, args.max_sequence_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    t_total = len(train_loader) * args.num_epochs
    warmup_steps = math.ceil(t_total * args.warmup_ratio)

    # build model
    model = MonoTransQuestModel(args.huggingface_model, tokenizer)

    N_EPOCHS = args.num_epochs
    optimizer = transformers.optimization.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )


    model.to(DEVICE)
    # logger.info(model)

    # train loop
    for epoch in range(N_EPOCHS):
        logger.info("Start epoch {}:".format(epoch+1))

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        valid_loss, cm, valid_acc, val_prec, val_rec, valid_f1 = evaluate(model, valid_loader)
        logger.info("Validation tp: {}, fn: {}, fp: {}, tn: {}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
        logger.info("Validation loss: {:.3f}, acc: {:.3f}, F1: {:.3f}".format(valid_loss, valid_acc, valid_f1))
    
    logger.info(model)



if __name__ == "__main__":

    parser = cfargparse.ArgParser()
    parser.add(
        "-c",
        "--config",
        type=str,
        is_config_file=True,
        help="Path to the configuration file",
    )
    parser.add(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data."
    )
    parser.add(
        "--valid_data",
        type=str,
        required=True,
        help="Path to the the val data"
    )
    parser.add(
        "--output_dir",
        type=str,
        required=True,
        help="Dir to save model",
    )
    parser.add(
        "--huggingface_model",
        type=str,
        default="xlm-roberta-base",
        help="The name of the pretrained model",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Dimension of custom layers",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=64,
        help="Max sequence length to use in transformer, use -1 to default to transformer maximum",
    )
    parser.add(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add("--train_batch_size", type=int, default=16, help="Batch size")
    parser.add("--valid_batch_size", type=int, default=4, help="Batch size")
    parser.add("--lr", type=float, default=2e-5, help="Learning Rate")
    parser.add("--warmup_ratio", type=float, default=0.2, help="Learning Rate")
    parser.add("--ner", type=bool, default=False, help="Use NER data or not")
    parser.add("--tox", type=bool, default=False, help="Use NER data or not")


    args = parser.parse_args()

    output_dir = args.output_dir + datetime.now().strftime('%Y_%m_%d')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    logger = setup_logger("train_log", os.path.join(output_dir, "ner_hidden_states.log"))
    logger.info("\n" + "=" * 30 + "Start training" + "=" * 30)
    logger.info(parser.format_values())
    logger.info("\nlr: {}\n".format(args.lr))
    train(args)
