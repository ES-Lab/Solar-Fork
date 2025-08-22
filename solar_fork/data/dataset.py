import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SeqDataset:
    def __init__(self, path, seq_length=56):
        self.seq_length = seq_length
        # Load the dataset
        self.df = pd.read_csv(path)
        self.df.set_index(self.df.columns[0], inplace=True)
        # Normalize the dataset
        self.X_scaled, self.y_scaled = self.normalize_df()
        # Create sequences
        self.X_seq, self.y_seq = self.create_sequences(seq_length)
        # Split the data
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data(0.8)

    def normalize_df(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # Features and target
        self.X = self.df.iloc[:, :-1].values  # All columns except the last one
        self.y = self.df.iloc[:, -1].values   # Last column
        X_scaled = self.scaler_X.fit_transform(self.X)
        y_scaled = self.scaler_y.fit_transform(self.y.reshape(-1, 1))
        return X_scaled, y_scaled

    def create_sequences(self, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(self.X_scaled) - seq_length):
            X_seq.append(self.X_scaled[i:i+seq_length])
            y_seq.append(self.y_scaled[i+seq_length])
        # print sequence shapes
        print(f"Sequence shapes: X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
        return np.array(X_seq), np.array(y_seq)

    def split_data(self, split_ratio):
        split_index = int(split_ratio * len(self.X_seq))
        X_train, X_test = self.X_seq[:split_index], self.X_seq[split_index:]
        y_train, y_test = self.y_seq[:split_index], self.y_seq[split_index:]
        # print split results
        print(f"Data split: Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
        # timestamps_test = self.df.index[self.seq_length + split_index:]