#%%
import json
from tqdm import tqdm
import pandas as pd

filename = "train"
users, items, types, ratings, timestamps = [], [], [], [], []

with open(f"./{filename}.jsonl", "r") as f:
    temp = f.readlines()


for i in tqdm(temp):
    i = json.loads(i)
    u = i["session"]
    for e in i["events"]:
        if e["type"] == "orders":
            users.append(u)
            items.append(e["aid"])
            types.append(e["type"])
            ratings.append(1)
            timestamps.append(e["ts"])

df = pd.DataFrame(
    {
        "user": users,
        "item": items,
        "rating": ratings,
        "timestamp": timestamps,
        "type": types,
    }
)


df.to_pickle(f"./{filename}.pkl")
