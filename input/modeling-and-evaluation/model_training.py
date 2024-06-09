import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def train_lgbm_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict
) -> tuple:
    """
    LightGBMモデルを訓練し、AUC-ROCスコアを返す

    Args:
    X_train (pd.DataFrame): 訓練用特徴量
    y_train (pd.Series): 訓練用ターゲット
    X_val (pd.DataFrame): バリデーション用特徴量
    y_val (pd.Series): バリデーション用ターゲット
    params (dict): モデルのハイパーパラメータ

    Returns:
    tuple: 訓練されたモデル、バリデーションデータに対するAUC-ROCスコア
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    gbm = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
    )

    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_val, y_pred)

    return gbm, auc
