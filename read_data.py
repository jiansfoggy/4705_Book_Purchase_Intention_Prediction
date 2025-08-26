import json
import pandas as pd

# 文件路径
review = "./data/Books.jsonl"
meta = "./data/meta_Books.jsonl"

# read JSONL and parse to Python dictionary line by line
overall_data = []
test_data = []
save_train = False
with open(review, 'r', encoding='utf-8') as fp:
    cnt = 0
    for line in fp:
        line = json.loads(line.strip())
        buy = line["verified_purchase"]
        if cnt < 200000:
            if len(line["title"])>0 and len(line["text"])>0 and len(str(buy))>0:
                comb_text = line["title"]+". "+line["text"]
                record = "Positive" if buy==1 else "Negative"
                overall_data.append({"text":comb_text,"bought":record})
                cnt += 1
                if cnt%20000==0:
                    print(cnt)
        elif cnt >= 200000 and cnt < 220001:
            if len(line["title"])>0 and len(line["text"])>0 and len(str(buy))>0:
                comb_text = line["title"]+". "+line["text"]
                record = "Positive" if buy==1 else "Negative"
                test_data.append({"text":comb_text,"bought":record})
                cnt += 1
                if cnt%5000==0:
                    print(cnt)

if save_train:
    df = pd.DataFrame(overall_data)
    df.to_csv("./data/review_data.csv", index=False)
    print("Column names:")
    print(df.columns.tolist())
else:
    with open("./data/test_data.json", "a", encoding="utf-8") as fout:
        json.dump(test_data, fout, ensure_ascii=False, indent=2)
    print("Wrote test_data.json with", len(test_data), "entries")


# train_data = []
# with open(meta, 'r', encoding='utf-8') as fp:
#     for idx, line in enumerate(fp):
#         if idx < 3001:
#             train_data.append(json.loads(line.strip()))
#             if idx%3000==0:
#                 print(idx)

# df = pd.DataFrame(train_data)
# df.to_csv("./data/meta_data.csv", index=False)
# print("Column names:")
# print(df.columns.tolist())

# python3 read_data.py
