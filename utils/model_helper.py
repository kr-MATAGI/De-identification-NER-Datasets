import os
import re

import torch
import copy
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification
#######################################################################################################################
# labels
NAVER_NE_MAP = {
    "O": 0,
    "B-PER": 1, "I-PER": 2, # 인물
    "B-FLD": 3, "I-FLD": 4, # 학문 분야
    "B-AFW": 5, "I-AFW": 6, # 인공물
    "B-ORG": 7, "I-ORG": 8, # 기관 및 단체
    "B-LOC": 9, "I-LOC": 10, # 지역명
    "B-CVL": 11, "I-CVL": 12, # 문명 및 문화
    "B-DAT": 13, "I-DAT": 14, # 날짜
    "B-TIM": 15, "I-TIM": 16, # 시간
    "B-NUM": 17, "I-NUM": 18, # 숫자
    "B-EVT": 19, "I-EVT": 20, # 사건사고 및 행사
    "B-ANM": 21, "I-ANM": 22, # 동물
    "B-PLT": 23, "I-PLT": 24, # 식물
    "B-MAT": 25, "I-MAT": 26, # 금속/암석/화학물질
    "B-TRM": 27, "I-TRM": 28, # 의학용어/IT관련 용어
    "X": 29, # special token
}

# regex
RE_paragraph_head = r"={2,5}\s[^a-zA-Z]+\s={2,5}"

# excpet label
NOT_NEED_TAGS = ["ANM", "PLT", "MAT"]

#### Method
def trained_model_load(tokenizer_name: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForTokenClassification.from_pretrained("../model")

    return tokenizer, model

def do_semi_auto_tagging(model, tokenizer, src_path: str, save_dir: str):
    src_file = open(src_path, mode="r", encoding="utf-8")
    save_file = open(save_dir+"/model_"+src_path.split("/")[-1], mode="w", encoding="utf-8")

    id2label = {v: k for k, v in NAVER_NE_MAP.items()}
    title = ""
    src_lines = src_file.readlines()
    for line_idx, src_line in enumerate(src_lines):
        if "\n" == src_line: # new doc
            title = ""
            continue
        if 0 >= len(title): # set title
            title = src_line.strip()
            continue

        if 0 == (line_idx % 1000):
            print(f"{line_idx} Processing, {src_path.split('/')[-1]}")

        text = src_line.strip()
        if ("분류:" in text) or ("틀:" in text) or (re.match(RE_paragraph_head, text)):
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)[0]

        conv_preds = list(id2label[x] for x in preds)
        tokens = tokens[1:-1]
        conv_preds = conv_preds[1:-1]

        new_tokens = []
        new_preds = []
        for tok, prd in zip(tokens, conv_preds):
            if "##" in tok:
                back_word = new_tokens[-1]
                back_word += tok.replace("##", "")
                new_tokens[-1] = back_word
            else:
                new_tokens.append(tok)
                if "TIM" == prd.split("-")[-1]:
                    prd = prd.split("-")[0]
                    prd += "-DAT"
                new_preds.append(prd if prd.split("-")[-1] not in NOT_NEED_TAGS else "O")
        assert len(new_tokens) == len(new_preds)

        save_file.write(title + "\n")
        save_file.write(text + "\n")
        for tok, prd in zip(new_tokens, new_preds):
            save_file.write(tok + "\t" + prd + "\n")
        save_file.write("\n")

    src_file.close()
    save_file.close()

def do_semi_auto_specific_word(model, tokenizer, src_path: str, save_path: str):
    src_file = open(src_path, mode="r", encoding="utf-8")
    save_file = open(save_path, mode="w", encoding="utf-8")

    id2label = {v: k for k, v in NAVER_NE_MAP.items()}
    src_iter = iter(src_file.readlines())
    while True:
        line = next(src_iter, None)
        if line is None:
            break
        if "\n" == line:
            continue
        title = copy.deepcopy(line.strip().replace("\n", ""))
        split_text = copy.deepcopy(next(src_iter, None).strip().replace("\n", "").split(". "))

        if (0 >= len(title)) or (0 >= len(split_text)):
            continue

        for text in split_text:
            text += "."
            text = text.strip()
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=2)[0]

            conv_preds = list(id2label[x] for x in preds)
            tokens = tokens[1:-1]
            conv_preds = conv_preds[1:-1]

            new_tokens = []
            new_preds = []
            for tok, prd in zip(tokens, conv_preds):
                if "##" in tok:
                    back_word = new_tokens[-1]
                    back_word += tok.replace("##", "")
                    new_tokens[-1] = back_word
                else:
                    new_tokens.append(tok)
                    if "TIM" == prd.split("-")[-1]:
                        prd = prd.split("-")[0]
                        prd += "-DAT"
                    new_preds.append(prd if prd.split("-")[-1] not in NOT_NEED_TAGS else "O")
            assert len(new_tokens) == len(new_preds)

            save_file.write(title + "\n")
            save_file.write(text + "\n")
            for tok, prd in zip(new_tokens, new_preds):
                save_file.write(tok + "\t" + prd + "\n")
            save_file.write("\n")



if "__main__" == __name__:
    # model
    tokenizer, model = trained_model_load(tokenizer_name="monologg/koelectra-base-v3-discriminator",
                                          model_dir="../model")
    model.eval()

    do_semi_auto_filter_data = False
    if do_semi_auto_filter_data:
        # semi-auto tagging using by model
        target_dir = "../data/filter"
        save_dir = "../data/model_output"
        target_file_list = os.listdir(target_dir)

        for f_idx, file_name in enumerate(target_file_list):
            src_path = target_dir + "/" + file_name
            do_semi_auto_tagging(model, tokenizer, src_path=src_path, save_dir=save_dir)

    is_do_semi_auto_specific_word = True
    if is_do_semi_auto_specific_word:
        src_path = "../data/specific/merge_email.txt"
        save_path = "../data/specific/model_email.txt"
        do_semi_auto_specific_word(model, tokenizer, src_path=src_path, save_path=save_path)