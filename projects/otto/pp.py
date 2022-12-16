#%%
import sys

sys.path.append("../..")
from src.configs import DATACLASS_PATH, OUTPUT_PATH, DATA_PATH
import pickle
import json
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Literal


def f(i):
    for e in i["events"]:
        event_dict[e["type"]]["session"].append(s)
        event_dict[e["type"]]["aid"].append(e["aid"])
        event_dict[e["type"]]["type"].append(e["type"])
        event_dict[e["type"]]["rating"].append(1)
        event_dict[e["type"]]["ts"].append(e["ts"])


def json2csv(
    filename: str, types: List[str] = ["clicks", "carts", "orders"]
) -> Dict[str, pd.DataFrame]:
    with open(DATA_PATH / f"{filename}.jsonl", "r") as f:
        temp = f.readlines()

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
            event_dict[e["type"]]["session"].append(s)
            event_dict[e["type"]]["aid"].append(e["aid"])
            event_dict[e["type"]]["type"].append(e["type"])
            event_dict[e["type"]]["rating"].append(1)
            event_dict[e["type"]]["ts"].append(e["ts"])

    for e in event_dict:
        df = pd.DataFrame(event_dict[e])
        event_dict[e] = df
        # df.to_pickle(f"./otto_{filename}_{e}.pkl")

    return event_dict


def check_item_in(items_small, items_big) -> float:
    if len(items_small) == 0:
        # print("No items in small")
        return -1
    intersection = list(set(items_small) & set(items_big))
    not_in_big = list(set(items_small) - set(items_big))
    coverage = len(intersection) / len(items_small)
    # print(len(not_in_big), coverage)
    return coverage


def check_coverage(events_df: Dict[str, pd.DataFrame]) -> None:
    """
    # clicks vs carts
    Not coverage users 31655
    coverage rate 0.981047078491801
    # carts vs orders
    Not coverage users 1927
    coverage rate 0.9920645052361088
    """
    users = events_df["carts"].session.unique()
    users_num = len(users)
    print("Totall user", users_num)
    not_coverage_users_num = 0
    for s in tqdm(users):
        s_clicks = (
            events_df["carts"][events_df["carts"].session == s].aid.unique().tolist()
        )
        s_carts = (
            events_df["orders"][events_df["orders"].session == s].aid.unique().tolist()
        )
        coverage = check_item_in(s_carts, s_clicks)

        if coverage != 1 and coverage != 0 and coverage != -1:
            not_coverage_users_num += 1

    print("Not coverage users", not_coverage_users_num)
    print("coverage rate", (users_num - not_coverage_users_num) / users_num)


#%%
events_df = json2csv("train")
check_coverage(events_df)
#%%
for e in events_df:
    events_df[e].to_pickle(DATA_PATH / f"otto_train_{e}.pkl")


# %%
# items_big = (
#     events_df["clicks"][events_df["clicks"].session == 12899782].aid.unique().tolist()
# )
# items_small = (
#     events_df["carts"][events_df["carts"].session == 12899782].aid.unique().tolist()
# )
# intersection = list(set(items_small) & set(items_big))
# not_in_big = list(set(items_small) - set(items_big))
# # print(intersection)
# print(not_in_big)
# items_small

# %%
# Check session which has more than 3 times of clicks
session_counts = events_df["clicks"].session.value_counts()
session_counts_3 = session_counts[session_counts >= 3]
print(
    len(session_counts),
    len(session_counts_3),
    len(session_counts_3) / len(session_counts),
)

# %%
for type in ["clicks", "carts", "orders"]:
    session_counts = events_df[type].session.value_counts()
    session_counts_3 = session_counts[session_counts >= 3]
    print(
        len(session_counts),
        len(session_counts_3),
        len(session_counts_3) / len(session_counts),
    )
