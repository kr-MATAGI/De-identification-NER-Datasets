import os

def check_sentences_count(src_path: str):
    if not os.path.exists(src_path):
        print(f"ERR - Not Existed: {src_path}")
        return

    sent_count = 0
    with open(src_path, mode="r", encoding="utf-8") as src_file:
        src_iter = iter(src_file.readlines())
        while True:
            title = next(src_iter, None)
            if title is None:
                break
            sent_count += 1
            title = title.replace("\n", "")
            sent = next(src_iter, None).replace("\n", "")
            token_list = []
            while True:
                token = next(src_iter, None)
                if "\n" == token or token is None:
                    break

    print(f"Total sentence counts: {sent_count}")

### MAIN ###
if "__main__" == __name__:
    check_sentences_count(src_path="../data/merge/원본_merge.txt")