import numpy as np


class LogRegression():

    def __init__(self, X, lr=1e-4):
        # (with defaults) as you see fit
        self.lr = lr
        self.m = np.zeros(X.shape[1])
        self.b = 0
        pass

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
            dm = -(1 / n) * np.dot(X.T, (y - y_pred))
            db = -(1 / n) * np.sum(y - y_pred)

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
        return 1 / (1 + np.exp(-(np.dot(X, self.m) + self.b)))
    
    def get_error_distribution(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.abs(y - y_pred)
