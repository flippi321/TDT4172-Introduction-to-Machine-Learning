import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error


class EnsembleRegressor():

    def __init__(self):
        # (with defaults) as you see fit
        self.m = None
        self.b = 0
        pass

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def rmsle(self, y_true, y_pred):
        return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

    def fit(self, X, y, lr=1e-2, epoch=1000, verbose=False, plot_loss=False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = np.array(X)
        y = np.array(y)
        n = len(y)

        self.m = np.random.randn(X.shape[1]) * 0.01

        for i in range(epoch):
            y_pred = self.predict(X)

            error = self.rmse(y, y_pred)

            # Compute gradients
            dm = -(2 / n) * np.dot(X.T, (y - y_pred))
            db = -(2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.m -= lr * dm
            self.b -= lr * db

            if verbose and (i % max(1, (epoch//10)) == 0):
                print(f"Epoch {i}: RMSLE={error:.4f}")


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        #raise NotImplementedError("The predict method is not implemented yet.")
        #X = np.array(X)
        return np.dot(X, self.m) + self.b

    def get_formula(self, dec=3):
        """
        Returns the formula of the linear regression model
        as a string in the form "y = mx + b"
        """
        return f"y = {round(self.m, dec)}x + {round(self.b, dec)}"
    
    def get_error_distribution(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.abs(y - y_pred)
