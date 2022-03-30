import logging
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

#######################################################################################################################
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

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger


if "__main__" == __name__:
    logger = init_logger()

    # config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # config = ElectraConfig.from_pretrained(model_name_or_path,
    #                                        num_labels=len(NAVER_NE_MAP.keys()),
    #                                        id2label={str(i): label for i, label in enumerate(NAVER_NE_MAP.keys())},
    #                                        label2id={label: i for i, label in enumerate(NAVER_NE_MAP.keys())})

    # model
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForTokenClassification.from_pretrained("../model")
    model.to(device)
    model.eval()


    inputs = tokenizer("안녕하세요, 최재훈 입니다.")
    outputs = model(inputs)