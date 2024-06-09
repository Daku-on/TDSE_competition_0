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

## TODO

|No.|Status|Name|Detail|Due Date|URL|
|---|---|---|---|---|---|
| 1 | Done | subの仕方を公開 | 参加のハードルを下げるため、簡単なモデルでsubする方法を公開する | 20240602 | N/A |
| 2 | Done | EDA notebookを作成 | 参加のハードルを下げるため、簡単なEDAも公開する | 20240602 | N/A |
| 3 | New | WBSをひく | とりあえずTODOの優先順位だけでもつける | 20240611 | N/A |
| 4 | New | CV vs. LB | trainとtestは分布が違いそうなのでcross val.は必須そう | N/A | N/A |
| 5 | Done | up/down sampling | 20240601参照 | N/A | N/A |
| 6 | New | 顧客のクラスタリング | おそらくいくつかクラスタがあるはずなので、仮説を検証しつつ必要ならモデルを分ける | N/A | N/A |
| 7 | New | モデル解釈 | 効いている特徴量の抽出と、顧客への説明 | N/A | N/A |

## Log

### 20240601

- 開始
- とりあえず imbalance binary dataなのですべて0埋めして提出
  - public LBは0.5だった。つまり半分が0, 半分が1というテストデータなので、trainとかなり分布を変えているはず。up/down samplingは試す価値がありそう。
- LightGBMで「動くだけ」のモデル、nb000を全体に向けて公開。参加者増えてくれ...
  
### 20240602

- ロマンティックウォーリアー強いですねぇ
- EDA用の関数をちまちま作っている。強いモデルはooooさんにお任せや！

### 20240608

- Fitbit Charge 5が届いた
- EEEDDDYYYにちょっとだけ返事してあとは休み。
- CustomerIdの使い方なんだよなぁ、たぶん。
- CustomerId, Surname, Geographyが同じでExited=1の人は1, あとは0で出したら0.5 -> 0.504に上がった。とはいえ割合に対してスコアの上昇度が低いのでこれをすべきかは微妙

### 20240609

- 行けるやろと思ったら体調死んだわ
- と言いつつとりあえずGenderとGeographyをone-hot encodingして0.884
  - down samplingした。本当はカラムの分布も見てサンプリングしたかったが実装が無理でした。結果は0.884 (nb002)
  - optunaで殴ればMakinoさん超えられそうな気がするのでやってみたが、結果は0.885 (nb003)。これたぶんLGBM単独モデルだと限界が近いよねぇ。
- 来週はクラスタリングしてモデル分けてみたい。