import pandas as pd
import numpy as np
import boto3
import io
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    TimeDistributed, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict


# -------------------------------------------------------
# Helper: Remove outliers using quantile thresholds
# -------------------------------------------------------
def remove_outliers(df, cols, low=0.05, high=0.95):
    out = df.copy()
    for c in cols:
        ql, qh = out[c].quantile(low), out[c].quantile(high)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out

