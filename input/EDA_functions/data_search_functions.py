"""
functions to search for data in the pd.DataFrame
"""
import numpy as np
import pandas as pd
from typing import Optional


def find_duplicates(
    df: pd.DataFrame,
    column_name: str
) -> pd.DataFrame:
    """
    Find and display rows with duplicated values in the specified column using groupby.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column name to check for duplicates.

    Returns:
        pd.DataFrame: Dataframe with duplicated rows.
    """
    # Group by the specified column and filter groups with more than one entry
    grouped_df = df.groupby(column_name).filter(lambda x: len(x) > 1)
    return grouped_df.sort_values(by=column_name)
