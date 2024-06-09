import pandas as pd
from sklearn.utils import resample


def downsampling(
        df: pd.DataFrame,
        target_column: str,
        majority_query: str,
        random_seed: int = 42
) -> pd.DataFrame:
    """
    不均衡データをダウンサンプリングし、すべてのカラムの分布を元データと一致させたデータフレームを返す

    Args:
    df (pd.DataFrame): 入力データフレーム
    target_column (str): ターゲットカラムの名前
    majority_query (str): マイノリティクラスを抽出するためのクエリ
    random_seed (int): ランダムシード

    Returns:
    pd.DataFrame: ダウンサンプリングされ、分布が調整されたデータフレーム
    """

    # クラスごとのデータ数の確認
    print(df[target_column].value_counts())  # target_columnの値の分布を出力する

    # マイノリティクラスとマジョリティクラスに分割
    df_majority = df.query(majority_query)  # クエリを使ってマイノリティクラスを抽出
    df_minority = df.drop(df_majority.index)  # 残りをマジョリティクラスとする

    # マジョリティクラスのdownsampling
    df_majority_downsampled = resample(
        df_majority,  # ダウンサンプリングするデータフレーム
        replace=False,  # サンプリング時に重複を許さない
        n_samples=len(df_minority),  # マイノリティクラスと同じ数にする
        random_state=random_seed  # 再現性のためにランダムシードを固定
    )

    # downsampledデータの結合
    df_downsampled = pd.concat(
        [df_majority_downsampled, df_minority]
    )  # ダウンサンプル後のマジョリティとマイノリティを結合

    return df_downsampled
