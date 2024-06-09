import pandas as pd
from sklearn.utils import resample


def downsample_and_match_distributions(
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
    majority_query (str): マジョリティクラスを選択するためのクエリ

    Returns:
    pd.DataFrame: ダウンサンプリングされ、分布が調整されたデータフレーム
    """

    # クラスごとのデータ数の確認
    print(df[target_column].value_counts())  # target_columnの値の分布を出力する

    # マイノリティクラスとマジョリティクラスに分割
    df_majority = df.query(majority_query)
    df_minority = df.drop(df_majority.index)

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

    # 全カラムの分布を元データと一致させる関数
    def match_all_distributions(
            df_downsampled: pd.DataFrame,
            df_original: pd.DataFrame,
            columns: pd.Index,
    ) -> pd.DataFrame:

        # 各カラムに対して処理を行う
        for column in columns:
            # 元データの分布を計算
            original_dist = df_original[column].value_counts(normalize=True)
            # ダウンサンプルデータの分布を計算
            sampled_dist = df_downsampled[column].value_counts(normalize=True)

            # 補正係数を計算
            correction_factor = original_dist / sampled_dist
            # 欠損値を0で埋める
            correction_factor = correction_factor.fillna(0)

            # 補正後のサンプルを格納するリスト
            corrected_samples = []

            # 各カテゴリに対して処理を行う
            for category, factor in correction_factor.items():
                # 該当カテゴリの行を抽出
                category_samples = df_downsampled[
                    df_downsampled[column] == category
                ]
                # 補正後のサンプル数を計算
                n_samples = int(len(category_samples) * factor)
                # サンプルをリサンプリング
                corrected_samples.append(
                    category_samples.sample(
                        n_samples,
                        replace=True,
                        random_state=random_seed,
                    )
                )

            # 補正後のサンプルを結合
            df_downsampled = pd.concat(corrected_samples)

        # 補正後のデータフレームを返す
        return df_downsampled

    # すべてのカラムの分布を元データと一致させる
    # target_column以外の全カラムを取得
    columns_to_match = df.columns.drop(target_column) 
    df_downsampled_corrected = match_all_distributions(
        df_downsampled,
        df,
        columns_to_match
    )  # 分布を一致させる関数を適用

    # 結果の確認
    # target_columnの分布を確認
    print(df_downsampled_corrected[target_column].value_counts())
    for column in columns_to_match:  # 各カラムの分布を確認
        print(df_downsampled_corrected[column].value_counts(normalize=True))

    # 補正後のデータフレームを返す
    return df_downsampled_corrected
