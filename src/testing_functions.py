"""
this file is for testing the flow and functions 
will be deleted later
"""
import pandas as pd
import numpy as np
import datatable as dt
from logger import logger
from utils import reduce_memory_usage, determine_usecase_type


def read_data_train(df):
    train = dt.fread(df).to_pandas().drop("id", axis=1)
    train = reduce_memory_usage(train)
    return train


def read_data_test(df):
    train = dt.fread(df).to_pandas().drop("id", axis=1)
    train = reduce_memory_usage(train)
    return train


train = "../datasets_samples/wisconsiv_binary_classification.csv"
train = read_data_train(train)

use_case_type = determine_usecase_type(3)
