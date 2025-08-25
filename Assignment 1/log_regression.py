import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class LogRegression():

    def __init__(self, n_features=1):
        # (with defaults) as you see fit
        self.m = np.random.randn(n_features) * 0.01
        self.b = np.random.randint(0, 10) * 0.01
        pass

    def fit(self, X, y, lr=1e-4, epoch=1000):
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
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)

                # Compute gradients
                dm = np.dot(xi.T, (y_pred - yi))
                db = (y_pred - yi)

                # Update parameters
                self.m -= lr * dm
                self.b -= lr * db

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
        return 1 / (1 + np.exp(-(np.dot(X, self.m) + self.b)))

    def measure_accuracy(self, test_X, test_y, tresh = 0.5, plot_roc=False):
        # Predicted probabilities
        pred_probs = self.predict(test_X)
        # Binary predictions at a custom treshold
        pred_labels = (pred_probs >= tresh).astype(int)

        # Accuracy
        acc = accuracy_score(test_y, pred_labels)

        # ROC AUC
        auc = roc_auc_score(test_y, pred_probs)

        if plot_roc:
            fpr, tpr, _ = roc_curve(test_y, pred_probs)
            plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()

        return {"accuracy": acc, "roc_auc": auc}

    def get_error_distribution(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.abs(y - y_pred)
    

    def get_error_distribution(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.abs(y - y_pred)
