import re
import os
import copy
from typing import List

'''
    Rule 1
    POS 
        - [가-힣]+학과 -> B-POS
        - 문장에 '전공' 이 있고, [가-힣]+학
'''
def do_rule_1(sent: str, lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    if re.search(r"[ㄱ-힣]+학과", ret_lhs):
        ret_is_conv = True
        ret_rhs = "B-POS"
    elif ("전공" in sent) and re.search(r"[$가-힣]+학$", ret_lhs) and ("대학" not in ret_lhs):
        ret_is_conv = True
        ret_rhs = "B-POS"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 2
    학적 상태
    POS
        - 졸업, 재학
'''
def do_rule_2(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    target_word = ["졸업", "재학", "전학", "휴학", "복학", "제적", "학사", "석사", "박사"]
    target_filter = list(filter(lambda x: True if x in ret_lhs else False, target_word))
    if 0 < len(target_filter) and "EVT" not in ret_rhs: # except '졸업식'
        ret_is_conv = True
        ret_rhs = "B-POS"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 3
    HEC
        - 문장에 투병이란 단어가 있고 아래의 정규 표현식에 해당
            1) [ㄱ-힣]*암
            2) [ㄱ-힣]+증세
            3) [ㄱ-힣]+증
            4) [ㄱ-힣]+병
            5) 뇌졸중
'''
def do_rule_3(sent: str, lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    target_re = [r"[ㄱ-힣]*암", r"[ㄱ-힣]+증세", r"[ㄱ-힣]+증", r"[ㄱ-힣]+병", "뇌졸중"]
    target_filter = list(filter(lambda x: True if re.search(x, ret_lhs) else False, target_re))
    if (0 < len(target_filter)) and ("투병" in sent) and ("투병" not in ret_lhs) and ("지병" not in ret_lhs) \
            and ("PER" not in ret_rhs) and ("ORG" not in ret_rhs):
        ret_is_conv = True
        ret_rhs = "B-HEC"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 4
    HEC
        - 혈액형(A형, B형, O형, AB형)
'''
def do_rule_4(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False
    
    target_word = ["A형", "B형", "O형", "AB형"]
    target_filter = list(filter(lambda x: True if x in ret_lhs else False, target_word))
    if 0 < len(target_filter):
        ret_is_conv = True
        ret_rhs = "B-HEC"
    
    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 5
    PIV
        - 문장에 "종교", "신자", "신앙" 이라는 단어가 있고
        - [ㄱ-힣]+교 라는 단어가 있다. 
'''
def do_rule_5(sent: str, lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    target_in_sent = ["종교", "신자", "신앙"]
    target_in_sent_filter = list(filter(lambda x: True if x in sent else False, target_in_sent))
    if 0 < len(target_in_sent_filter):
        if (re.search(r"[ㄱ-힣]+교", ret_lhs)) and ("종교" not in ret_lhs):
            ret_is_conv = True
            ret_rhs = "B-PIV"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 6
    PIV
        - 가족관계에 대한 표현
            1) [ㄱ-힣]*아버지, [ㄱ-힣]+머니, [ㄱ-힣]*아들, [ㄱ-힣]*딸
            2) 장남, 장녀, 차남, 차녀, 막내
            3) 결혼, 이혼
'''
def do_rule_6(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    target_re = [r"[가-힣]*아버지", r"[가-힣]+머니", r"[가-힣]*아들", r"[가-힣]*딸"]
    target_re_filter = list(filter(lambda x: True if re.search(x, ret_lhs) else False, target_re))
    target_word = ["장남", "장녀", "차남", "차녀", "막내", "결혼", "이혼", "파혼",
                   "동생", "오빠", "언니", "삼촌", "자녀", "손자", "손녀"]
    target_word_filter = list(filter(lambda x: True if x in ret_lhs else False, target_word))
    target_rel_re = [r"[0-9]+(남|녀)"]
    target_rel_filter = list(filter(lambda x: True if re.search(x, ret_lhs) else False, target_rel_re))

    if 0 < len(target_re_filter): # regex
        ret_is_conv = True
        ret_rhs = "B-PIV"
    elif 0 < len(target_word_filter): # search word
        ret_is_conv = True
        ret_rhs = "B-PIV"
    elif 0 < len(target_rel_filter):
        ret_is_conv = True
        ret_rhs = "B-PIV"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    창작물(AFW) 등에 포함되는 괄호에 태그 제거
        - <, 《, (, 「
        - >, 》, ), 」
'''
def do_rule_7(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    target_barket = ["<", "《", "(", "「", ">", "》", ")", "」"]
    target_filter = list(filter(lambda x: True if x in ret_lhs else False, target_barket))
    if 0 < len(target_filter):
        ret_is_conv = True
        ret_rhs = "O"

    return ret_lhs, ret_rhs, ret_is_conv

'''
    Rule 8
        - "I"로 태깅이 되어있지만 앞에 "B"가 없을 경우, "B"로 수정
'''
def do_rule_8(token_list: List[str], t_idx: int, lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)

    if 0 == t_idx:
        if "I-" in ret_rhs:
            ret_rhs = ret_rhs.replace("I-", "B-")
    else:
        if "I-" in ret_rhs:
            prev_token_sp = token_list[t_idx-1].split("\t")
            if "O" == prev_token_sp[-1]:
                ret_rhs = ret_rhs.replace("I-", "B-")

    return ret_lhs, ret_rhs
'''
    Rule 9
        - 문장 첫 번째에 '*'이 있는데 태깅이 되었을 경우
'''
def do_rule_9(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)

    if ("*" == ret_lhs) and ("O" != ret_rhs):
        ret_rhs = "O"

    return ret_lhs, ret_rhs

'''
    Rule 10
        - "~장애"
'''
def do_rule_10(lhs: str, rhs: str):
    ret_lhs = copy.deepcopy(lhs)
    ret_rhs = copy.deepcopy(rhs)
    ret_is_conv = False

    if re.search(r"[가-힣]+장애", ret_lhs):
        ret_is_conv = True
        ret_rhs = "B-HEC"

    return ret_lhs, ret_rhs, ret_is_conv

def convert_tag_use_regex(src_path: str, save_path: str):
    print(f"[convert_tag_use_regex] START - Path: {src_path}")

    if not os.path.exists(src_path):
        print(f"[convert_tag_use_regex] ERR - Not Exist {src_path}")
        return

    # save file
    save_file = open(save_path, mode="w", encoding="utf-8")

    total_sent_cnt = 0
    src_file = open(src_path, mode="r", encoding="utf-8")
    src_iter = iter(src_file.readlines())
    while True:
        title = next(src_iter, None)
        if title is None:
            break
        total_sent_cnt += 1
        title = title.replace("\n", "")
        sent = next(src_iter, None).replace("\n", "")
        token_list = []
        while True:
            token = next(src_iter, None)
            if "\n" == token or token is None:
                break
            token_list.append(token.replace("\n", ""))

        # apply rule and write file
        save_file.write(title + "\n")
        save_file.write(sent + "\n")
        for t_idx, token in enumerate(token_list):
            lhs = token.split("\t")[0]
            rhs = token.split("\t")[-1]

            is_conv = False
            lhs, rhs, is_conv = do_rule_1(sent, lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_2(lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_3(sent, lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_4(lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_5(sent, lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_6(lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_7(lhs, rhs)
            if not is_conv:
                lhs, rhs, is_conv = do_rule_10(lhs, rhs)

            lhs, rhs = do_rule_8(token_list, t_idx, lhs, rhs)
            if 0 == t_idx:
                lhs, rhs = do_rule_9(lhs, rhs)

            # write new tagging
            save_file.write(lhs+"\t"+rhs+"\n")
        save_file.write("\n")

    src_file.close()
    save_file.close()
    print(f"[convert_tag_use_regex] count: {total_sent_cnt} END - Path: {save_path}")

### MAIN ###
if "__main__" == __name__:
    print("[tag_regerx.py] START !")

    src_path = "./data/additional/tag_conv/conv_model_person4.txt"
    save_path = "./data/additional/regex/re_model_person4.txt"

    convert_tag_use_regex(src_path, save_path)