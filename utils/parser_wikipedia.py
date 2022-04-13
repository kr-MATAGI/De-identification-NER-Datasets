import os
import pickle
import re

import xml.etree.ElementTree as ET
from multiprocessing import Process

#### Global
g_person_doc_cnt = 0
g_company_doc_cnt = 0

#### Method
def read_kor_wiki_xml(src_path: str= ""):
    if not os.path.exists(src_path):
        print(f"[parser_wikipedia][read_kor_wiki_xml] ERR - NOT Existed : {src_path}")

    document = ET.parse(src_path)
    root = document.getroot()

    doc_cnt = 0
    for page in root.iter(tag="page"):
        doc_cnt += 1
        title = page.find("title").text
        revision = page.find("revision")
        text = revision.find("text").text

        if 0 == (doc_cnt % 50000):
            print(f"doc_cnt: {doc_cnt}, title: {title}")

        yield title, text

    print(f"[parser_wikipedia][read_kor_wiki_xml] Complete : {src_path}")

def is_person_category(doc_text):
    # [[분류:살아있는 사람]]
    person_re_1 = r"\[\[분류:[가-힣]+ 사람\]\]"
    person_re_2 = r"\[\[분류:[가-힣0-9]+ 사망\]\]"
    for t_line in doc_text.split("\n"):
        if re.search(person_re_1, t_line) or re.search(person_re_2, t_line):
            return True
    return False

def is_company_category(doc_text):
    # [[분류:한국 증권거래소 상장 기업]]
    company_re = r"\[\[분류:[가-힣0-9]+ 기업\]\]"
    for t_line in doc_text.split("\n"):
        if re.findall(company_re, t_line):
            return True
    return False

def parse_kor_wiki_xml(src_path: str, file_idx: int, save_dir: str):
    global g_person_doc_cnt
    global g_company_doc_cnt

    person_doc_list = []
    company_doc_list = []

    # 분류
    for doc_title, doc_text in read_kor_wiki_xml(src_path):
        if (None == doc_text) or (0 >= len(doc_text)):
            continue
        if is_person_category(doc_text):
            person_doc_list.append((doc_title, doc_text))
        elif is_company_category(doc_text):
            company_doc_list.append((doc_title, doc_text))
    
    # 저장
    person_save_path = save_dir + "/person_" + str(file_idx) + ".pkl"
    company_save_path = save_dir + "/company_" + str(file_idx) + ".pkl"
    with open(person_save_path, mode="wb") as person_pkl:
        pickle.dump(person_doc_list, person_pkl)
    with open(company_save_path, mode="wb") as company_pkl:
        pickle.dump(company_doc_list, company_pkl)

    g_person_doc_cnt += len(person_doc_list)
    g_company_doc_cnt += len(company_doc_list)
    print(f"[parser_wikipedia][parse_kor_wiki_xml] Save - {file_idx}")
    print(f"global count - [person: {g_person_doc_cnt}, company: {g_company_doc_cnt}]")


if "__main__" == __name__:
    print("[parser_wikipedia][main] ---START !")

    is_parse_kor_wiki = True
    if is_parse_kor_wiki:
        src_dir = "../data"
        save_dir = "../data/classify"

        procs = []
        src_file_list = list(filter(lambda x: True if ".xml" in x else False, os.listdir(src_dir)))
        for f_idx, file_name in enumerate(src_file_list):
            src_path = src_dir + "/" + file_name
            proc = Process(target=parse_kor_wiki_xml, args=(src_path, f_idx + 1, save_dir))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
