import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import pickle
from sklearn.model_selection import train_test_split

import model_training as my_model_training


def optimize_lgbm_hyperparameters(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        seed: int = 42,
        model_path: str = "../working/best_lgbm_model.pkl"
) -> optuna.Study:
    """
    Optunaを使用してLightGBMのハイパーパラメータを最適化する

    Args:
    X_train (pd.DataFrame): 訓練用特徴量
    y_train (pd.Series): 訓練用ターゲット
    n_trials (int): Optunaの試行回数
    seed (int): ランダムシード

    Returns:
    optuna.Study: 最適化結果を含むOptunaのStudyオブジェクト
    """

    # データを訓練用とバリデーション用に分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    # Optunaによるハイパーパラメータ最適化
    def objective(trial):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',  # 評価指標をAUCに設定
            'seed': seed,  # モデル全体のランダムシードを固定
            'random_state': seed,  # モデルの特定のプロセスに対するランダムシードを固定
            'bagging_seed': seed,  # バギングのランダムシードを固定
            'feature_fraction_seed': seed,  # 特徴量サンプリングのランダムシードを固定
            'verbosity': -1,  # 出力を抑制
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        _, auc = my_model_training.train_lgbm_model(
            X_train, y_train, X_val, y_val, params
        )

        return auc

    # Optunaの最適化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # 最適なハイパーパラメータを取得
    best_params = study.best_trial.params

    # 最適なハイパーパラメータで最終モデルを訓練
    best_model, _ = my_model_training.train_lgbm_model(
        X_train, y_train, X_val, y_val, best_params
    )

    # 最終モデルをpickleで保存
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # 最適なハイパーパラメータを表示
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study
