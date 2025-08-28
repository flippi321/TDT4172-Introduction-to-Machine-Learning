import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


class LogRegression():

    def __init__(self, degrees=1):
        self.m = None
        self.b = np.random.randint(0, 10) * 0.1 
        self.poly = PolynomialFeatures(degree=degrees)
        pass

    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)),
                        np.exp(-z) / (1 + np.exp(z)))

    def polynomial_preprocessing(self, X):
        return self.poly.fit_transform(np.array(X))

    def fit(self, X, y, lr=1e-4, epoch=1000, verbose=False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = self.polynomial_preprocessing(X)
        y = np.array(y)
        n = len(y)

        self.m = np.random.randn(X.shape[1]) * 0.01

        for i in range(epoch):
            y_pred = self.predict(X)

            error = y - y_pred

            # Compute gradients
            cost = -(1/n) * np.sum(y*np.log(y_pred+1e-15) + (1-y)*np.log(1-y_pred+1e-15))
            dm = (1 / n) * np.dot(X.T, error)
            db = np.mean(error)

            # Update parameters
            self.m += lr * dm
            self.b += lr * db

            if verbose and (i % max(1, (epoch//10)) == 0):
                print(f"Epoch {i}: cost={cost:.4f}")

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

    def measure_accuracy(self, test_X, test_y, tresh=0.5, plot_roc=False):
        # Predicted probabilities
        test_X = self.polynomial_preprocessing(test_X)
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
