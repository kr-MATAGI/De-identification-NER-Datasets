import os
import copy
import re

####################################################################################
convert_tag_dict = {
    "FLD": "O", "CVL": "POS"
}

blood_type = ["A형", "B형", "O형", "AB형"]
####################################################################################

def convert_tag(src_path: str, save_path: str):
    print(f"[convert_tag] Convert Start - {src_path}")

    save_file = open(save_path, mode="w", encoding="utf-8")
    with open(src_path, mode="r", encoding="utf-8") as src_file:
        for line in src_file.readlines():
            if "\t" in line:
                sp_line = line.split("\t")
                lhs_text = sp_line[0]
                rhs_text = sp_line[-1]

                rhs_sp = rhs_text.split("-")
                is_blood_type = list(filter(lambda x: True if x in lhs_text else False, blood_type))
                if rhs_sp[-1].replace("\n", "") in convert_tag_dict.keys():
                    front_bio = rhs_sp[0]
                    back_tag = (convert_tag_dict[rhs_sp[-1].replace("\n", "")] + "\n")
                    if "O\n" != back_tag:
                        rhs_text = (front_bio + "-" + back_tag)
                    else:
                        rhs_text = back_tag
                elif 0 < len(is_blood_type):
                    rhs_text = "B-HEC\n"

                save_file.write(lhs_text + "\t" + rhs_text)
            else:
                save_file.write(line)
    save_file.close()
    print(f"[convert_tag] Complete - {save_path}")

#### Main
if "__main__" == __name__:
    target_category = "person"
    src_dir = "../data/model_output"
    save_dir = "../data/tag_conv"
    src_file_list = list(filter(lambda x: True if target_category in x else False, os.listdir(src_dir)))

    for src_idx, src_file in enumerate(src_file_list):
        src_path = src_dir + "/" + src_file
        save_path = save_dir + "/conv_" + target_category + str(src_idx+1) + ".txt"
        convert_tag(src_path, save_path)
