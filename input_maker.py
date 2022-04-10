import numpy as np
import copy
import os
from tag_def import DE_IDENT_TAG, DE_IDENT_ZIP

import torch
from transformers import ElectraTokenizer

def read_src_liens(src_path: ""):
    with open(src_path, mode="r", encoding="utf-8") as src_file:
        src_lines = src_file.readlines()
        src_iter = iter(src_lines)
        read_cnt = 0
        while True:
            title = next(src_iter, None)
            if title is None:
                break
            read_cnt += 1

            title = title.replace("\n", "")
            sentence = next(src_iter).replace("\n", "")
            segment_list = []
            while True:
                segment = next(src_iter, None)
                if ("\n" == segment) or (segment is None):
                    break
                segment = segment.replace("\n", "")
                segment_list.append(segment)

            ret_items = DE_IDENT_ZIP(title=title,
                                     sent=sentence,
                                     segment_list=copy.deepcopy(segment_list))
            yield ret_items

def make_npy(src_path: str, save_path: str, model_name: str, max_len: int):
    npy_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "labels": []
    }

    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    for read_items in read_src_liens(src_path):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        for seg in read_items.segment_list:
            lhs, rhs = seg.split("\t")
            token_list = tokenizer(lhs)

            token_len = len(token_list["input_ids"][1:-1])
            if max_len - 2 < (token_len + len(input_ids)):
                break
            else:
                input_ids.extend(token_list["input_ids"][1:-1])
                token_type_ids.extend(token_list["token_type_ids"][1:-1])
                attention_mask.extend(token_list["attention_mask"][1:-1])

                for t_idx in range(token_len):
                    if 0 == t_idx:
                        try:
                            labels.append(DE_IDENT_TAG[rhs])
                        except:
                            print(read_items.sent, "\n", lhs, seg)
                            exit()
                    else:
                        labels.append(DE_IDENT_TAG["X"])
        # end loop, read_items.segment_list

        input_ids.insert(0, 2) # [CLS]
        labels.insert(0, -100)
        token_type_ids.insert(0, 0)
        attention_mask.insert(0, 0)

        if max_len-1 > len(input_ids):
            pad_list = [0 for x in range(max_len-len(input_ids)-1)]
            x_list = [DE_IDENT_TAG["X"] for _ in range(max_len-len(input_ids)-1)]
            input_ids.extend(pad_list)
            labels.extend(x_list)
            token_type_ids.extend(pad_list)
            attention_mask.extend(pad_list)

        input_ids.append(3) # [SEP]
        labels.append(-100)
        token_type_ids.append(0)
        attention_mask.append(0)

        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(token_type_ids)
        assert len(input_ids) == len(attention_mask)

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["attention_mask"].append(attention_mask)
    # end loop, read_src_liens()

    # save_file
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    print(f"size: {npy_dict['input_ids'].shape}")

    np.save(save_path+"/input_ids", npy_dict["input_ids"])
    np.save(save_path + "/labels", npy_dict["labels"])
    np.save(save_path + "/token_type_ids", npy_dict["token_type_ids"])
    np.save(save_path + "/attention_mask", npy_dict["attention_mask"])


def split_npy_input(src_dir: str):
    print(f"[split_npy_input] src_dir: {src_dir}")

    train_dir = src_dir + "/train"
    test_dir = src_dir + "/test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # load src_*.npy
    src_input_ids = np.load(src_dir+"/input_ids.npy")
    src_labels = np.load(src_dir+"/labels.npy")
    src_attention_mask = np.load(src_dir+"/attention_mask.npy")
    src_token_type_ids = np.load(src_dir+"/token_type_ids.npy")

    total_sent_size = src_input_ids.shape[0]
    train_idx_list = np.random.choice(total_sent_size, 1200, False)
    test_idx_list = []
    for idx in range(total_sent_size):
        if idx not in train_idx_list:
            test_idx_list.append(idx)
    print(f"train_idx_list.size: {len(train_idx_list)}")
    print(f"test_idx_list.size: {len(test_idx_list)}")

    # make train/test *.npy dataset
    train_input_ids = [src_input_ids[idx] for idx in range(total_sent_size) if idx in train_idx_list]
    train_labels = [src_labels[idx] for idx in range(total_sent_size) if idx in train_idx_list]
    train_token_type_ids = [src_token_type_ids[idx] for idx in range(total_sent_size) if idx in train_idx_list]
    train_attention_mask = [src_attention_mask[idx] for idx in range(total_sent_size) if idx in train_idx_list]

    test_input_ids = [src_input_ids[idx] for idx in range(total_sent_size) if idx in test_idx_list]
    test_labels = [src_labels[idx] for idx in range(total_sent_size) if idx in test_idx_list]
    test_token_type_ids = [src_token_type_ids[idx] for idx in range(total_sent_size) if idx in test_idx_list]
    test_attention_mask = [src_attention_mask[idx] for idx in range(total_sent_size) if idx in test_idx_list]

    train_input_ids = np.array(train_input_ids)
    train_labels = np.array(train_labels)
    train_token_type_ids = np.array(train_token_type_ids)
    train_attention_mask = np.array(train_attention_mask)

    test_input_ids = np.array(test_input_ids)
    test_labels = np.array(test_labels)
    test_token_type_ids = np.array(test_token_type_ids)
    test_attention_mask = np.array(test_attention_mask)

    print(f"np.shape - train: {train_input_ids.shape}, test: {test_input_ids.shape}")

    # save train/test *.npy
    np.save(train_dir+"/input_ids", train_input_ids)
    np.save(train_dir+"/labels", train_labels)
    np.save(train_dir+"/token_type_ids", train_token_type_ids)
    np.save(train_dir+"/attention_mask", train_attention_mask)

    np.save(test_dir+"/input_ids", test_input_ids)
    np.save(test_dir+"/labels", test_labels)
    np.save(test_dir+"/token_type_ids", test_token_type_ids)
    np.save(test_dir+"/attention_mask", test_attention_mask)

### MAIN ###
if "__main__" == __name__:

    do_make_all_input = False
    if do_make_all_input:
        src_path = "./data/merge/test_regex_merge_프로토타입.txt"
        save_path = "./npy"
        make_npy(src_path=src_path,
                 save_path=save_path,
                 model_name="monologg/koelectra-base-v3-discriminator",
                 max_len=128)

    do_split_made_input = True
    if do_split_made_input:
        src_dir = "./npy"
        split_npy_input(src_dir=src_dir)