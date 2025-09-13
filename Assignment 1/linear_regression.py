import numpy as np


class LinearRegression():

    def __init__(self, ):
        # (with defaults) as you see fit
        self.m = 0
        self.b = 0
        self.loss = []
        pass

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def fit(self, X, y, lr=1e-4, epoch=1000, log_loss=False):
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
            error = y - y_pred

            # Compute gradients
            dm = -(2 / n) * np.sum(X * error)
            db = -(2 / n) * np.sum(error)

            # Update parameters
            self.m -= lr * dm
            self.b -= lr * db

            if(log_loss):
                self.loss.append(np.sum(error))

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