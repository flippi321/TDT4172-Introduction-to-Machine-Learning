import numpy as np


class LinearRegression():

    def __init__(self, lr=1e-4):
        # (with defaults) as you see fit
        self.lr = lr
        self.m = 0
        self.b = 0
        pass

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def fit(self, X, y, epoch=1000):
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

        for _ in range(epoch):
            y_pred = self.predict(X)

            # Compute gradients
            dm = -(2 / n) * np.sum(X * (y - y_pred))
            db = -(2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.m -= self.lr * dm
            self.b -= self.lr * db

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
        return X * self.m + self.b

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
