import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures


class EnsembleRegressor():

    def __init__(self, degrees=0, threshold=0):
        # (with defaults) as you see fit
        self.poly = PolynomialFeatures(degree=degrees)
        self.threshold = threshold

        self.m_small, self.b_small = None, 0
        self.m_big, self.b_big = None, 0
        pass

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def polynomial_preprocessing(self, X):
        return self.poly.fit_transform(np.array(X))

    def rmsle(self, y_true, y_pred):
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

    def fit(self, X, y, lr=1e-2, epoch=1000, verbose=False, plot_loss=False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = self.polynomial_preprocessing(X)
        y = np.array(y)
        mask_small = y <= self.threshold
        mask_big = y > self.threshold

        self.m_small, self.b_small = self.train_section(epoch, X, y[mask_small], lr, verbose)
        self.m_big, self.b_big = self.train_section(epoch, X, y[mask_big], lr, verbose)
        
    def train_section(self, epoch, X, y, lr, verbose):
        n = len(y)
        m = np.random.randn(X.shape[1]) * 0.01
        b = 0

        for i in range(epoch):
            y_pred = np.dot(X, m) + b

            error = self.rmsle(y, y_pred)

            # Compute gradients
            dm = -(2 / n) * np.dot(X.T, (y - y_pred))
            db = -(2 / n) * np.sum(y - y_pred)

            # Update parameters
            m -= lr * dm
            b -= lr * db

            if verbose and (i % max(1, (epoch // 10)) == 0):
                print(f"Epoch {i}: RMSLE={error:.4f}")
        
        return m, b

    def predict(self, X, transform=False):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        if transform:
            X = self.polynomial_preprocessing(X)
        
        return np.dot(X, self.m) + self.b

    def get_error_distribution(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.abs(y - y_pred)
