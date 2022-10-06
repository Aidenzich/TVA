# LightRec
此專案用於建立方便使用的推薦系統模型，
基本上架構應採用比較寬鬆的設計，
結合 Pytorch Lighting 簡化 pytorch 框架的複雜度

## 資料前處理
- 容許資料格式
    | userid | itemid | rating | timestemp |
    |-|-|-|-|
    | 1 | 2 | 1.333 |  |
    | 2 | 3 | 1.003 |  |

## Datasets 設計
我們使用 cornac 套件作為資料源
基本上分成兩種：
- SeqDataset
- CfDateset

## Models
- [x] Bert4Rec
- [ ] SASRec
- [ ] Carca
- [ ] VAECF
- [ ] BiVAECF

## Docker
### Run
```
docker compose up -d
```

### Often use
```
scp -r ~/External/lightRec ikmnew:~/
```