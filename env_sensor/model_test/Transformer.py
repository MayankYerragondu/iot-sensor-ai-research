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


# -------------------------------------------------------
# Helper: Create sliding-window sequences
# -------------------------------------------------------
def create_sequences(data, n_steps=10):
    """
    Converts normalized features into supervised sequences.

    X: past n_steps timesteps of features
    y: same sequence’s temperature, humidity, lux
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i:i+n_steps, :3])  # predict all timesteps
    return np.array(X), np.array(y)


# -------------------------------------------------------
# Transformer Encoder-Decoder Model
# -------------------------------------------------------
def build_transformer(input_shape, heads=4, ff_dim=64):
    """
    Simple Transformer model for sequence-to-sequence regression.
    - Multi-head attention over input
    - Feed-forward dense block
    - TimeDistributed(Dense) to predict features per timestep
    """
    inp = Input(shape=input_shape)

    # Self-attention
    attn = MultiHeadAttention(num_heads=heads, key_dim=input_shape[1])(inp, inp)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inp + attn)

    # Feed-forward block
    ff = Dense(ff_dim, activation="relu")(out1)
    ff = Dense(input_shape[1])(ff)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff)

    # Output: 3 features per timestep
    out = TimeDistributed(Dense(3))(out2)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

