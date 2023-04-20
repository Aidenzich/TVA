import random


def neg_sample(item_set, item_size) -> int:
    """
    random sample an item id that is not in the user's interact history
    """
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item
