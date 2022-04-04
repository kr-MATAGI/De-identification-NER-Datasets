import os

def check_sentences_count(src_path: str):
    if not os.path.exists(src_path):
        print(f"ERR - Not Existed: {src_path}")
        return

    sent_count = 0
    with open(src_path, mode="r", encoding="utf-8") as src_file:
        for line in src_file.readlines():
            if "\t" not in line.strip():
                sent_count += 1

    print(f"Total sentence counts: {sent_count//2}")

### MAIN ###
if "__main__" == __name__:
    check_sentences_count(src_path="../data/merge/merge_person.txt")