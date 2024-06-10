import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Self


class TargetEncoder:
    """
    Target Encoder for categorical features with optional smoothing and holdout encoding.

    Attributes:
        min_samples_leaf (int): Minimum samples to take category average into account.
        smoothing (float): Smoothing effect to balance categorical average vs prior.
    """

    def __init__(
        self,
        min_samples_leaf: int = 1,
        smoothing: float = 1.0
    ):
        """
        Initialize the Target Encoder with specified parameters.

        Args:
            min_samples_leaf (int): Minimum samples to take category average into account.
            smoothing (float): Smoothing effect to balance categorical average vs prior.
        """
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.target_means = {}

    def fit(
        self,
        X: pd.Series,
        y: pd.Series,
        is_smoothing: Optional[bool] = True,
        is_holdout: Optional[bool] = True,
    ) -> Self:
        """
        Fit the encoder according to the given training data.

        Args:
            X (pd.Series): Input feature (categorical column).
            y (pd.Series): Target variable.
            is_smoothing (Optional[bool]): Apply smoothing if True.
            is_holdout (Optional[bool]): Apply holdout encoding if True.

        Returns:
            self
        """
        if is_holdout:
            self.target_means = self._calculate_holdout_target_means(X, y)
        elif is_smoothing:
            self.target_means = self._calculate_smooth_target_means(X, y)
        else:
            self.target_means = self._calculate_target_means(X, y)
        return self

    def transform(
        self,
        X: pd.Series,
    ) -> pd.Series:
        """
        Transform X according to the learned target encoding.

        Args:
            X (pd.Series): Input feature (categorical column).

        Returns:
            pd.Series: Transformed feature.
        """
        return X.map(self.target_means)

    def _calculate_target_means(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> dict:
        """
        Calculate the target mean for each category in the feature without smoothing.

        Args:
            X (pd.Series): Input feature.
            y (pd.Series): Target variable.

        Returns:
            dict: Mapping of category to target mean.
        """
        return y.groupby(X).mean().to_dict()

    def _calculate_smooth_target_means(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> dict:
        """
        Calculate the target mean for each category in the feature with smoothing.

        Args:
            X (pd.Series): Input feature.
            y (pd.Series): Target variable.

        Returns:
            dict: Mapping of category to smoothed target mean.
        """
        target_means = {}

        # Calculate global mean of the target
        global_mean = y.mean()

        # Calculate counts and means for each category
        stats = y.groupby(X).agg(['count', 'mean'])
        counts = stats['count']
        means = stats['mean']

        # Calculate smoothed means
        smoothing = 1 / (1 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
        smooth_means = global_mean * (1 - smoothing) + means * smoothing

        # Create a dictionary mapping each category to its smoothed mean
        target_means = smooth_means.to_dict()

        return target_means

    def _calculate_holdout_target_means(
        self,
        X: pd.Series,
        y: pd.Series,
        n_splits: int = 5,
        random_seed: int = 42
    ) -> dict:
        """
        Calculate the target mean for each category using holdout method.

        Args:
            X (pd.Series): Input feature.
            y (pd.Series): Target variable.
            n_splits (int): Number of splits for holdout method.

        Returns:
            dict: Mapping of category to holdout target mean.
        """
        target_means = pd.Series(index=X.index, dtype=float)
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed
        )

        for train_idx, holdout_idx in skf.split(X, y):
            X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
            y_train = y.iloc[train_idx]

            means = y_train.groupby(X_train).mean()
            target_means.iloc[holdout_idx] = X_holdout.map(means)

        return target_means.to_dict()


def target_encode(
    train_df: pd.DataFrame,
    categorical_column: str,
    target_column: str,
    is_smoothing: Optional[bool] = True,
    is_holdout: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Apply target encoding to a categorical column in the dataframe.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        categorical_column (str): The name of the categorical column to encode.
        target_column (str): The name of the target column.
        is_smoothing (Optional[bool]): Apply smoothing if True.
        is_holdout (Optional[bool]): Apply holdout encoding if True.

    Returns:
        pd.DataFrame: The dataframe with the target encoded column.
    """
    encoder = TargetEncoder()

    # Fit the encoder
    encoder.fit(
        train_df[categorical_column],
        train_df[target_column],
        is_smoothing,
        is_holdout
    )

    # Transform the column
    train_df[categorical_column + '_encoded'] = encoder.transform(train_df[categorical_column])

    return train_df


"""
Usage:

# サンプルデータの作成
data = {
    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'target': [1, 2, 1, 2, 3, 1]
}
train_df = pd.DataFrame(data)

# ターゲットエンコーディングの実行（ホールドアウトあり）
result_df_holdout = target_encode(train_df, 'category', 'target', is_smoothing=False, is_holdout=True)
print("Holdout Encoding:")
print(result_df_holdout)

# ターゲットエンコーディングの実行（スムージングあり）
result_df_smooth = target_encode(train_df, 'category', 'target', is_smoothing=True, is_holdout=False)
print("Smoothing Encoding:")
print(result_df_smooth)
"""