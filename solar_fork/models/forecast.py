from ..models.nn import *


class ForecastModel:
    def __init__(self, name, seq_length, input_dim, load_model=None):
        self.name = name
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.model = self.build_model()
        if load_model:
            self.load_model(load_model)

    def build_model(self):
        if self.name == "gru":
            return build_gru_model(self.seq_length, self.input_dim)
        elif self.name == "lstm":
            return build_lstm_model(self.seq_length, self.input_dim)
        elif self.name == "cnn":
            return build_cnn_model(self.seq_length, self.input_dim)
        elif self.name == "tcn":
            return build_tcn_model(self.seq_length, self.input_dim)
        elif self.name == "transformer":
            return build_transformer_model(self.seq_length, self.input_dim)
        elif self.name == "ann":
            return build_ann_model(self.seq_length, self.input_dim)
        else:
            raise ValueError(f"Unknown model type: {self.name}")
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        print(f"\n Training {self.name} model...")
        if self.name == "ann": X_train = X_train.reshape(X_train.shape[0], -1)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        return history

    def predict(self, X_test):
        if self.name == "ann": X_test = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = self.model.load_model(filepath)
