import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def plot_roc_and_calculate_auc(
    actuals_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    actuals_column: str,
    predictions_column: str,
) -> float:
    """
    ROC曲線を描画し、AUCを計算する関数。

    Parameters:
    actuals_df (pd.DataFrame): 実際の値が含まれるデータフレーム。
    predictions_df (pd.DataFrame): 予測値が含まれるデータフレーム。
    actuals_column (str): 実際の値が入っているカラム名。
    predictions_column (str): 予測値が入っているカラム名。

    Returns:
    float: AUC値。
    """
    # 実際の値と予測値を取り出す
    # 実際の値を1次元の配列に変換
    y_true = actuals_df[actuals_column].values.ravel()
    # 予測値を1次元の配列に変換
    y_scores = predictions_df[predictions_column].values.ravel()

    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # AUCの計算
    auc = roc_auc_score(y_true, y_scores)

    # ROC曲線のプロット
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='blue',
        lw=2,
        label=f'ROC curve (area = {auc:.2f})',
    )
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return auc
