import numpy as np
import pandas as pd


def loguniform(low=0, high=1, size=None, base=10):
    """Returns a number or a set of numbers from a log uniform distribution"""
    return np.power(base, np.random.uniform(low, high, size))


def encode_dates(df, column):
    df.copy()
    df[column + "_year"] = df[column].dt.year
    df[column + "_month"] = df[column].dt.month
    df[column + "_day"] = df[column].dt.day

    df[column + "_hour"] = df[column].dt.hour
    df[column + "_minute"] = df[column].dt.minute
    df[column + "_second"] = df[column].dt.second
    df = df.drop(column, axis=1)
    return df