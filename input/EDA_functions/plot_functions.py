"""
functions for plotting
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


def plot_distributions(
    df: pd.DataFrame,
    is_category_dict: Optional[dict[str, bool]] = None,
    title: str = "Data Distributions",
) -> None:
    """
    Plot the distributions of all columns in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        category_flag (Optional[dict[str, int]], optional): Dictionary indicating categorical variables (column name: 0 or 1). Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Data Distributions".
    """
    if is_category_dict is None:
        # Automatically recognize categorical variables (consider string or object types as categorical)
        is_category_dict = {
            col: int((df[col].dtype == "object") or (df[col].dtype == "string"))
            for col in df.columns
        }
    else:
        pass

    # Columns for numerical variables
    num_cols = [col for col, flag in is_category_dict.items() if flag == 0]
    # Columns for categorical variables
    cat_cols = [col for col, flag in is_category_dict.items() if flag == 1]

    # plot settings
    n_cols = 4  # Number of plots per row
    n_rows = (len(num_cols) + len(cat_cols)) // n_cols + ((len(num_cols) + len(cat_cols)) % n_cols > 0)  # Calculate required rows

    plt.figure(figsize=(20, n_rows * 5))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(
            df[col],  # Data for numerical variable
            bins=30,  # Number of bins
            kde=True,  # Display KDE (kernel density estimation) plot
        )
        plt.title(col)

    for i, col in enumerate(cat_cols, len(num_cols) + 1):
        subplot_ax = plt.subplot(n_rows, n_cols, i)  # subplot_ax に変更
        category_counts = df[col].value_counts(normalize=True) * 100  # Normalize counts to percentages
        sns.barplot(
            x=category_counts.index,
            y=category_counts.values,
            ax=subplot_ax
        )
        subplot_ax.yaxis.set_major_formatter(PercentFormatter())  # Set y-axis to display percentages
        plt.title(col)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
) -> None:
    """
    Plot a heatmap of correlations.

    Args:
        df (pd.DataFrame): The dataframe.
        title (str, optional): Title of the heatmap. Defaults to "Correlation Heatmap".
    """
    plt.figure(figsize=(15, 10))
    correlation_matrix_df = df.corr()
    sns.heatmap(
        correlation_matrix_df,  # Correlation matrix
        annot=True,  # Display values in cells
        fmt=".2f",  # Format of values
        cmap="coolwarm",  # Colormap
        vmin=-1,  # Minimum value
        vmax=1,  # Maximum value
    )
    plt.title(title, fontsize=20)
    plt.show() 