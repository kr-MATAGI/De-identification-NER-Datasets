import copy
import json
import os
import logging
import numpy as np
import random

import glob
import re
import argparse
from attrdict import AttrDict

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import ElectraConfig, get_linear_schedule_with_warmup, ElectraForTokenClassification

from tqdm import tqdm

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from tag_def import DE_IDENT_TAG, DE_IDENT_ZIP


################################################################################################################
def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" == args.device:
        torch.cuda.manual_seed_all(args.seed)

def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds),
            "recall": seqeval_metrics.recall_score(labels, preds),
            "f1": seqeval_metrics.f1_score(labels, preds),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }

def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds)

class De_Ident_Dataset(Dataset):
    def __init__(self, path: str=""):
        self.attention_mask = np.load(path+"/attention_mask.npy")
        self.input_ids = np.load(path+"/input_ids.npy")
        self.labels = np.load(path+"/labels.npy")
        self.token_type_ids = np.load(path+"/token_type_ids.npy")

        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "attention_mask": self.attention_mask[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx]
        }

        return items

################################################################################################################
##### GLOBAL
# Logger
logger = init_logger()

# Tensorboard
if not os.path.exists("./logs"):
    os.mkdir("./logs")
tb_writer = SummaryWriter("./logs")
################################################################################################################


################################################################################################################
######################################## Train / Eval ##########################################################
################################################################################################################
def train(args, model, train_dataset, dev_dataset, test_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # eps : 줄이기 전/후의 lr차이가 eps보다 작으면 무시한다.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # @NOTE: optimizer에 설정된 learning_rate까지 선형으로 감소시킨다. (스케줄러)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            model.train()
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "labels": batch["labels"].to(args.device)
            }

            if args.is_crf:
                log_likelihood, outputs = model(**inputs)
                loss = -1 * log_likelihood
            else:
                outputs = model(**inputs)
                loss = outputs[0]

            if 1 < args.n_gpu:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                (len(train_dataloader) <= args.gradient_accumulation_steps and \
                 (step + 1) == len(train_dataloader)
                ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                tb_writer.add_scalar("Loss/train", tr_loss / global_step, global_step)
                pbar.set_description("Train Loss - %.04f" % (tr_loss / global_step))

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save samples checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving samples checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info("  Epoch Done= %d", epoch)
        pbar.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, global_step=None, train_epoch=0):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    if None != global_step:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))

    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    eval_pbar = tqdm(eval_dataloader)
    for batch in eval_pbar:
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "labels": batch["labels"].to(args.device)
            }

            if args.is_crf:
                log_likelihood, outputs = model(**inputs)
                loss = -1 * log_likelihood
                eval_loss += loss.mean().item()
            else:
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        tb_writer.add_scalar("Loss/val_" + str(train_epoch), eval_loss / nb_eval_steps, nb_eval_steps)
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))

        if preds is None:
            if args.is_crf:
                preds = np.array(outputs)
                out_label_ids = inputs["labels"].detach().cpu().numpy() # 128, 128
            else:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            if args.is_crf:
                preds = np.append(preds, np.array(outputs), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    logger.info("  Eval End !")
    eval_pbar.close()

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    if not args.is_crf:
        preds = np.argmax(preds, axis=2)

    # nikl
    #labels = TTA_NE_tags.keys()

    # naver
    labels = DE_IDENT_TAG.keys()
    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    x_token_label_id = DE_IDENT_TAG["X"]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if (out_label_ids[i, j] != pad_token_label_id) and \
                    (out_label_ids[i, j] != x_token_label_id):
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

    result = f1_pre_rec(out_label_list, preds_list, is_ner=True)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list))  # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))

    return results

################################################################################################################
### MAIN
################################################################################################################
def main(cli_args):
    # Read config.json file and make args
    with open(cli_args.config_file) as config_file:
        args = AttrDict(json.load(config_file))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args)

    logger.info(f"Training/Evaluation parameters {args}")
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    # Config - naver ner labels
    config = ElectraConfig.from_pretrained(args.model_name_or_path,
                                           num_labels=len(DE_IDENT_TAG.keys()),
                                           id2label={str(i): label for i, label in enumerate(DE_IDENT_TAG.keys())},
                                           label2id={label: i for i, label in enumerate(DE_IDENT_TAG.keys())})

    # Model
    model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path, config=config)

    # GPU or CPU
    if 1 < torch.cuda.device_count():
        logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        args.n_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    # Load datasets
    train_dataset = De_Ident_Dataset(path=args.train_dir) if args.train_dir else None
    dev_dataset = De_Ident_Dataset(path=args.dev_dir) if args.dev_dir else None
    test_dataset = De_Ident_Dataset(path=args.test_dir) if args.test_dir else None

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = ElectraForTokenClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda key_with_step: (
                        "".join(re.findall(r'[^_]+_', key_with_step)),
                        int(re.findall(r"_\d+", key_with_step)[-1][1:])
                )):
                    f_w.write("{} = {}\n".format(key, str(results[key])))
            else:
                for key in sorted(results.keys()):
                    f_w.write("{} = {}\n".format(key, str(results[key])))

if "__main__" == __name__:
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_file", type=str, required=True)
    cli_args = cli_parser.parse_args()

    main(cli_args)

    # Tensorboard close
    tb_writer.close()