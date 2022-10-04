import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", Default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tune:
        tune_bert4rec()
    else:
        train_bert4rec()
