# Recsys Sequence Playground
這個repo用來統整我發論文所需要的比較對手，以及我論文主題的試驗場地

## Compare models
- [ ] Bert4Rec
- [ ] SASRec
- [ ] Carca

## Datasets
- 目前預計使用 cornac 作為資料集
- 預計使用的資料格式
    | userid | itemid | rating |
    |-|-|-|
    | 1 | 2 | 1.333 |
    | 2 | 3 | 1.003 |

## Docker
### Install
```
docker build -t recsys .
```

### Run
```
docker run -p 7777:7777 -p 6006:6006 --gpus all -v $PWD/data:/home/myuser/data recsys
```

## Tensorboard
```
tensorboard --logdir logs/lightning_logs
```