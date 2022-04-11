import copy
import pickle
import os
import re

### Regex
from wiki_syntax import WIKI_RE
from parser_wikipedia import read_kor_wiki_xml

RE_paragraph_head = r"={2}\s[^a-zA-Z]+\s={2}"
RE_infobox_open = r"\{\{"
RE_infobox_close = r"\}\}"
RE_table_open = r"\{\|"
RE_table_close = r"\|\}"

#### 문단 제목 - 첫 문단의 문단은 사용
person_paragraph_target = [
    # 1차 필터링 - 1500문장
    #"생애", "생활", "성장", "가족", "경력", 
    
    "평판", "개요", "학력", "데뷔", "기타", "에피소드",
    "시절", "평가", "이슈", "논란", "사건"
]

company_paragraph_target = [
    # 1차 필터링 - 1500문장
    #"개요", "역사", "사업", "규모", "실적",
    #"인수", "합병", "기본 정보"
]

#### Method
def extract_valid_text(text: str):
    ret_text = copy.deepcopy(text).strip()

    ret_text = re.sub(WIKI_RE.CITE.value, "", ret_text)

    ret_text = re.sub(WIKI_RE.REF.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.REF_2.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.REF_3.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.REF_4.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.REF_5.value, "", ret_text)

    ret_text = re.sub(WIKI_RE.COMMENT.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.PRE.value, "", ret_text)

    ret_text = re.sub(WIKI_RE.BR.value, "", ret_text)

    # Font Shape
    if re.search(WIKI_RE.FONT_SHAPE_5.value, ret_text):
        fontShape = re.search(WIKI_RE.FONT_SHAPE_5.value, ret_text).group(0)
        convertFontShape = fontShape.replace("'''''", "")
        ret_text = ret_text.replace(fontShape, convertFontShape)
    if re.search(WIKI_RE.FONT_SHAPE_3.value, ret_text):
        fontShape = re.search(WIKI_RE.FONT_SHAPE_3.value, ret_text).group(0)
        convertFontShape = fontShape.replace("'''", "")
        ret_text = ret_text.replace(fontShape, convertFontShape)
    if re.search(WIKI_RE.FONT_SHAPE_2.value, ret_text):
        fontShape = re.search(WIKI_RE.FONT_SHAPE_2.value, ret_text).group(0)
        convertFontShape = fontShape.replace("''", "")
        ret_text = ret_text.replace(fontShape, convertFontShape)

    # special character
    ret_text = re.sub(WIKI_RE.SPECIAL_CHAR.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.SUBP_SCRIPT.value, "", ret_text)

    # <> tag
    ret_text = re.sub(WIKI_RE.SPAN_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.MATH_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.SMALL_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.BIG_TAG.value, "", ret_text)

    ret_text = re.sub(WIKI_RE.ONLY_INCLUDE_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.INCLUDE_ONLY_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.NO_INCLUDE_TAG.value, "", ret_text)
    ret_text = re.sub(WIKI_RE.NO_WIKI_TAG.value, "", ret_text)

    # Redirect
    if re.search(WIKI_RE.REDIRECT.value, ret_text):
        corresStr = re.search(WIKI_RE.REDIRECT.value, ret_text).group(0)
        convertedStr = corresStr.replace("#넘겨주기 [[", "")
        convertedStr = convertedStr.replace("]]", "")
        ret_text = ret_text.replace(corresStr, convertedStr)

    # Free Link
    if re.search(WIKI_RE.FREE_LINK_ALT.value, ret_text):
        ret_text = re.sub(WIKI_RE.FREE_LINK_LHS.value, "", ret_text)
        ret_text = re.sub(WIKI_RE.FREE_LINK_CLOSED.value, "", ret_text)

    if re.search(WIKI_RE.FREE_LINK_BASIC.value, ret_text):
        ret_text = re.sub(WIKI_RE.FREE_LINK_OPEN.value, "", ret_text)
        ret_text = re.sub(WIKI_RE.FREE_LINK_CLOSED.value, "", ret_text)

    # External Link
    if re.search(WIKI_RE.EXT_LINK_ALT.value, ret_text):
        corresStr = re.search(WIKI_RE.EXT_LINK_ALT.value, ret_text).group(0)
        convertedStr = re.sub(WIKI_RE.EXT_LINK_ALT_LHS.value, "", corresStr)
        convertedStr = re.sub(r"\]", "", convertedStr)
        ret_text = ret_text.replace(corresStr, convertedStr)

    # Exception
    ret_text = ret_text.replace("[[", "")
    ret_text = ret_text.replace("<>", "")

    return ret_text

def extract_paragraph_from_doc(src_path: str, pkl_idx: int, save_dir: str, mode: str):
    load_list = []
    with open(src_path, mode="rb") as src_file:
        load_list = pickle.load(src_file)
    save_file = open(save_dir+"/"+mode+str(pkl_idx)+".txt", mode="w", encoding="utf-8")

    if "person" == mode:
        head_target_list = person_paragraph_target
    else:
        head_target_list = company_paragraph_target

    for d_idx, doc in enumerate(load_list):
        if 0 == (d_idx + 1) % 1000:
            print(f"{d_idx+1} Processing...")

        doc_title = doc[0]
        doc_text_split = doc[1].split("\n")
        filter_doc_text = []

        is_introduction = True
        is_need_info = False
        is_table = False
        for text_line in doc_text_split:
            if 0 >= len(text_line):
                continue
            if "각주" in text_line: # stop read page
                break

            if re.search(RE_infobox_open, text_line) and not re.search(RE_infobox_close, text_line):
                is_need_info = False
                is_introduction = False
                continue
            if not re.search(RE_infobox_open, text_line) and re.search(RE_infobox_close, text_line):
                is_need_info = True
                is_introduction = True
                continue

            if re.search(RE_table_open, text_line):
                is_table = True
            if re.search(RE_table_close, text_line):
                is_table = False
                continue
            if is_table:
                continue

            if re.match(RE_paragraph_head, text_line):
                is_introduction = False
                is_need_info = False
                for head_target in head_target_list:
                    if head_target in text_line:
                        is_need_info = True
                        break

            if is_introduction or is_need_info:
                valid_text = extract_valid_text(text_line)
                filter_doc_text.append(valid_text)

        # # save file, split filter data
        save_file.write(doc_title+"\n")
        for d_text in filter_doc_text:
            split_d_text_list = d_text.split(". ")
            for split_text in split_d_text_list:
                if 0 >= len(split_text.strip()):
                    continue
                if "." != split_text.strip()[-1] and "=" != split_text.strip()[-1]:
                    save_file.write(split_text.strip() + ".\n")
                else:
                    save_file.write(split_text.strip() + "\n")
        save_file.write("\n")
    save_file.close()

def extract_specific_target_sent(src_path: str, file_idx: int, save_dir: str, target_word: str):
    print(f"[extract_specific_target_sent] {file_idx} Start - {src_path}")
    # 분류
    save_list = []
    save_path = save_dir + "/" + target_word + str(file_idx) + ".txt"
    save_file = open(save_path, mode="w", encoding="utf-8")
    for doc_title, doc_text in read_kor_wiki_xml(src_path):
        if (None == doc_text) or (0 >= len(doc_text)):
            continue

        split_doc_text = doc_text.split("\n")
        filter_list = list(filter(lambda x: True if target_word in x else False, split_doc_text))
        if 0 < len(filter_list):
            valid_text = [extract_valid_text(x) for x in filter_list]
            save_file.write(doc_title + "\n")
            for save_text in valid_text:
                save_file.write(save_text + "\n")
            save_file.write("\n")
    save_file.close()
    print(f"[extract_specific_target_sent] {file_idx} Saved, {save_path}")


#### Main
if "__main__" == __name__:
    print("[semi_auto_maker][main] ----START ")

    is_classify_file = True
    if is_classify_file:
        classify_path = "../data/classify"
        filter_dir = "../data/filter"
        target = "person" # use person / company
        pkl_file_list = list(filter(lambda x: True if target in x else False, os.listdir(classify_path)))

        # Multi process
        for pkl_idx, pkl_name in enumerate(pkl_file_list):
            src_path = classify_path + "/" + pkl_name

            print(f"[START] {pkl_name}")
            extract_paragraph_from_doc(src_path=src_path, pkl_idx=(pkl_idx + 1), save_dir=filter_dir, mode=target)
            print(f"[END] {pkl_name}")

    is_extract_target = False
    if is_extract_target:
        src_dir = "../data"
        save_dir = "../data/specific"
        src_file_list = list(filter(lambda x: True if ".xml" in x else False, os.listdir(src_dir)))
        for f_idx, file_name in enumerate(src_file_list):
            src_path = src_dir + "/" + file_name
            extract_specific_target_sent(src_path, f_idx + 1, save_dir, target_word="행사")