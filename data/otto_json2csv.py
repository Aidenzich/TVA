#%%
import json
from tqdm import tqdm
import pandas as pd

filename = "test"

with open(f"./{filename}.jsonl", "r") as f:
    temp = f.readlines()

types = ["clicks", "carts", "orders"]

event_dict = {}
for t in types:
    event_dict[t] = {
        "session": [],
        "aid": [],
        "type": [],
        "rating": [],
        "ts": [],
    }


for i in tqdm(temp):
    i = json.loads(i)
    s = i["session"]
    for e in i["events"]:
        # if e["type"] == "orders":
        event_dict[e["type"]]["session"].append(s)
        event_dict[e["type"]]["aid"].append(e["aid"])
        event_dict[e["type"]]["type"].append(e["type"])
        event_dict[e["type"]]["rating"].append(1)
        event_dict[e["type"]]["ts"].append(e["ts"])


# df = pd.DataFrame(event_dict["orders"])
# print(df)
# df.to_pickle(f"./otto_{filename}_orders.pkl")
for e in event_dict:
    df = pd.DataFrame(event_dict[e])
    df.to_pickle(f"./otto_{filename}_{e}.pkl")

# %%
df

# %%
