#%%
"""
This script is used to check the distribution of the 'beauty.txt', 
Which is the dataset used in the S3-Rec paper.
The distribution of the dataset looks like the same as the 'ratings_Beauty_5.csv'
But current is difficult to restore the id mapping from the 'beauty.txt' to the 'ratings_Beauty_5.csv'
"""


from tqdm import tqdm

with open("./beauty.txt") as f:
    data = f.read()


lines = data.split("\n")
result_dict = {}


df_dict = {
    "user": [],
    "item": [],
    "rate": [],
    "time": [],
}

item_count = {}
total_user_num = 0
average_seq_len = 0

for line in tqdm(lines):
    items = line.split()
    user = items[0]
    user_seq = list(map(int, items[1:]))
    average_seq_len = average_seq_len + len(user_seq)
    total_user_num += 1
    for item in user_seq:
        if item not in item_count:
            item_count[item] = 1
        else:
            item_count[item] += 1

        df_dict["user"].append(user)
        df_dict["item"].append(item)
        df_dict["rate"].append(1)
        df_dict["time"].append(0)

    result_dict[user] = user_seq

average_seq_len = average_seq_len / total_user_num

print(average_seq_len)
print(item_count)
print(result_dict)


#%%
import pandas as pd

df = pd.read_csv("ratings_Beauty_5.csv")
df.columns
print(df.user_id.nunique())
print(df.item_id.nunique())

print(df.item_id.value_counts().tolist())
print(sorted(list(item_count.values()), reverse=True))

#%%
df.item_id.value_counts().to_dict()
