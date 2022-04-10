import numpy as np

from tag_def import DE_IDENT_TAG

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def trained_model_load(tokenizer_name: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForTokenClassification.from_pretrained("../model")

    return tokenizer, model

### MAIN ###
if "__main__" == __name__:
    print("[output_checker] START")


    # model
    # tokenizer, model = trained_model_load(tokenizer_name="monologg/koelectra-base-v3-discriminator",
    #                                       model_dir="./model")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForTokenClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model.eval()

    test_np_dir = "./npy/test"
    test_input_ids_np = np.load(test_np_dir+"/input_ids.npy")
    test_labels_ids_np = np.load(test_np_dir+"/labels.npy")
    test_attention_mask_np = np.load(test_np_dir+"/attention_mask.npy")
    test_token_type_ids_np = np.load(test_np_dir+"/token_type_ids.npy")

    id2label = {v:k for k, v in DE_IDENT_TAG.items()}
    data_size = test_input_ids_np.shape[0]
    for data_idx in range(data_size):
        inputs = {
            "input_ids": torch.tensor(test_input_ids_np[data_idx], dtype=torch.long),
            "attention_mask": torch.tensor(test_attention_mask_np[data_idx], dtype=torch.long),
            "token_type_ids": torch.tensor(test_token_type_ids_np[data_idx], dtype=torch.long)
        }
        print(inputs["input_ids"])

        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"])
        logits = outputs.logits
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)[0]

        conv_preds = list(id2label[x] for x in preds)
        labels = list(id2label[x] for x in test_labels_ids_np[data_idx])
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids_np[data_idx])

        print(f"{data_idx}, {tokens}")
        print(f"{data_idx}, {labels}")
        print(f"{data_idx}, {conv_preds}")
