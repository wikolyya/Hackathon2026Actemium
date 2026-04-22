class LSTMWrapper:

    def __init__(self, trainer, forecaster, config):
        self.trainer = trainer
        self.forecaster = forecaster
        self.config = config

        self.result = None

    def fit(self, X_train, y_train, X_val, y_val):

        self.result = self.trainer(X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, **self.config)

        return self

    def predict(self, X_last, n_steps):

        return self.forecaster(model=self.result[0], val=X_last, scaler=self.result[1], saisonalite=self.config["seq_len"], n_steps=n_steps)