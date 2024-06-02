# TDSE_competition_0

第0回TDSEコンペ用レポジトリ (終了後公開)

[Kaggle日記という戦い方](https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068) 参考

## Dataset

|Name|Detail|Ref|
|---|---|---|

## Paper

|No.|Status|Name|Detail|Date|Url|
|---|---|---|---|---|---|

## train.csv columns

|name|Explanation|
|----|----|

## Log

### 20240601

- 開始
- とりあえず imbalance binary dataなのですべて0埋めして提出
  - 0.5だった。つまり半分が0, 半分が1というテストデータなので、trainとかなり分布を変えているはず。up/down samplingは試す価値がありそう。
- LightGBMで「動くだけ」のモデル、nb000を全体に向けて公開。参加者増えてくれ...
  
### 20240602

- ロマンティックウォーリアー強いですねぇ
- EDA用の関数をちまちま作っている。
