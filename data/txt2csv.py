#%%
from tqdm import tqdm
with open("./beauty.txt") as f:
    data = f.read()

# 將資料轉換成 list of lines
lines = data.split("\n")

# 初始化一個空字典
result_dict = {}


df_dict = {
    "user": [],
    "item": [],
    "rate": [],
    "time": [],
}

# 遍歷每一行資料
for line in tqdm(lines):
    # 切割每一行，以空白字元分隔
    items = line.split()

    # 取得第一個元素作為 key
    user = items[0]

    # 取得第二個元素以後的元素作為 value，並轉換成整數
    user_seq = list(map(int, items[1:]))

    for item in user_seq:
        df_dict["user"].append(user)
        df_dict["item"].append(item)
        df_dict["rate"].append(1)
        df_dict["time"].append(0)

    # 將 key-value 加入字典
    result_dict[user] = user_seq



print(result_dict)

#%%
import pandas as pd

df = pd.DataFrame(df_dict)
df 

#%%
df.to_csv("./beauty.csv", index=False)