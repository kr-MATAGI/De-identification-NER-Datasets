import copy

import numpy as np

from tag_def import DE_IDENT_TAG

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def trained_model_load(tokenizer_name: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    return tokenizer, model

def check_test_datasets(test_dir: str, model_dir: str, save_path: str):
    save_file = open(save_path, mode="w", encoding="utf-8")
    save_file.write("word\tlabel\tpred\n")

    # model
    tokenizer, model = trained_model_load(tokenizer_name="monologg/koelectra-base-v3-discriminator",
                                          model_dir=model_dir)
    model.eval()

    test_input_ids_np = np.load(test_dir + "/input_ids.npy")
    test_labels_ids_np = np.load(test_dir + "/labels.npy")
    test_attention_mask_np = np.load(test_dir + "/attention_mask.npy")
    test_token_type_ids_np = np.load(test_dir + "/token_type_ids.npy")

    id2label = {v: k for k, v in DE_IDENT_TAG.items()}
    data_size = test_input_ids_np.shape[0]
    for data_idx in range(data_size):
        inputs = {
            "input_ids": torch.tensor([test_input_ids_np[data_idx]], dtype=torch.long),
            "attention_mask": torch.tensor([test_attention_mask_np[data_idx]], dtype=torch.long),
            "token_type_ids": torch.tensor([test_token_type_ids_np[data_idx]], dtype=torch.long)
        }
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"])
        logits = outputs.logits
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)[0]

        conv_preds = list(id2label[x] for x in preds)
        labels = list(id2label[x] if -100 != x else "O" for x in test_labels_ids_np[data_idx])
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids_np[data_idx])

        print(f"{data_idx + 1}, {tokens}")
        print(f"{data_idx + 1}, {conv_preds}")
        print(f"{data_idx + 1}, {labels}\n")

        tokens = tokens[1:-1]
        labels = labels[1:-1]
        conv_preds = conv_preds[1:-1]

        new_tokens = []
        new_labels = []
        new_preds = []
        for tok, la, prd in zip(tokens, labels, conv_preds):
            if "[SEP]" == tok:
                break
            elif "##" in tok:
                back_word = new_tokens[-1]
                back_word += tok.replace("##", "")
                new_tokens[-1] = back_word
            else:
                new_tokens.append(tok)
                new_labels.append(la)
                new_preds.append(prd)

        for tok, la, prd in zip(new_tokens, new_labels, new_preds):
            if la != prd:
                save_file.write(tok + "\t" + la + "\t" + prd + "<!>" + "\n")
            else:
                save_file.write(tok + "\t" + la + "\t" + prd + "\n")
        save_file.write("\n")
    save_file.close()


### MAIN ###
if "__main__" == __name__:
    print("[output_checker] START")

    do_check_test_npy = False
    if do_check_test_npy:
        check_test_datasets(test_dir="./npy/test", model_dir="./model_base", save_path="./output_check.txt")

    do_check_sent = True
    if do_check_sent:
        tokenizer, model = trained_model_load(tokenizer_name="monologg/koelectra-base-v3-discriminator",
                                              model_dir="./model-base")
        model.eval()

        input_str = "그 사람의 취미는 음악 감상이다."
        inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512)

        outputs = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"])
        logits = outputs.logits
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)[0]

        id2label = {v: k for k, v in DE_IDENT_TAG.items()}
        conv_preds = list(id2label[x] for x in preds)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        conv_preds = conv_preds[1:-1]

        new_tokens = []
        new_preds = []
        for tok, prd in zip(tokens, conv_preds):
            if "[SEP]" == tok:
                break
            elif "##" in tok:
                back_word = new_tokens[-1]
                back_word += tok.replace("##", "")
                new_tokens[-1] = back_word
            else:
                new_tokens.append(tok)
                new_preds.append(prd)

        for tok, prd in zip(new_tokens, new_preds):
                print(tok + "\t" + prd)