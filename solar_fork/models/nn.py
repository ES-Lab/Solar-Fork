
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Conv1D, Flatten, Dropout, MaxPooling1D, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

# --------------------------------------------------------
# 2. Model Definitions
# --------------------------------------------------------

# RNN model
def build_rnn_model(seq_length, input_dim, units=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        SimpleRNN(units, input_shape=(seq_length, input_dim)),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# GRU model
def build_gru_model(seq_length, input_dim, units=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        GRU(units, input_shape=(seq_length, input_dim)),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# LSTM model
def build_lstm_model(seq_length, input_dim, LSTM_units=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        LSTM(LSTM_units, return_sequences=True, input_shape=(seq_length, input_dim)),
        LSTM(LSTM_units),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# CNN model
def build_cnn_model(seq_length, input_dim, filters=64, kernel_size=3, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Conv1D(filters, kernel_size=kernel_size, activation='relu', input_shape=(seq_length, input_dim)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters*2, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# TCN model
def build_tcn_model(seq_length, input_dim, filters=64, kernel_size=3, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=1, activation='relu', padding='causal', input_shape=(seq_length, input_dim)),
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=2, activation='relu', padding='causal'),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Transformer model
def build_transformer_model(seq_length, input_dim, embed_dim=56, num_heads=2, ff_dim=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001):
    inputs = Input(shape=(seq_length, input_dim))
    
    # Positional Encoding can be added, but here we keep simple (check Transformer LSTM.ipynb for positional enconding)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    x = LayerNormalization()(x)
    
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(embed_dim)(x)
    x = LayerNormalization()(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# ANN model
def build_ann_model(seq_length, input_dim, dense_units=64, dropout_rate=0.3, learning_rate=0.001):  
    model = Sequential()
    model.add(Dense(dense_units, activation='relu', input_shape=(seq_length*input_dim,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


