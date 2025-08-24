import json
import pandas as pd

# 文件路径
file = "./data/Books.jsonl"

# 逐行读取 JSONL 并解析为 Python 字典
data = []
with open(file, 'r', encoding='utf-8') as fp:
    for line in fp:
        data.append(json.loads(line.strip()))

# 转换为 DataFrame
df = pd.DataFrame(data)

# 打印列名
print("Column names:")
print(df.columns.tolist())

