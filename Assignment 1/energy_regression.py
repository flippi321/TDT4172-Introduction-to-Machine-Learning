import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error


class EnsembleRegressor():

    def __init__(self, degrees=0):
        self.m = None
        self.b = np.random.randint(0, 10) * 0.1
        self.use_poly = True if (degrees != 0) else False
        self.poly = PolynomialFeatures(degree=degrees, include_bias=False)
        pass

    def polynomial_preprocessing(self, X):
        return self.poly.fit_transform(np.array(X))

    def rmsle(self, y_true, y_pred):
        """ Root Mean Squared Logarithmic Error """
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def rmse(self, y_true, y_pred):
        """ Root Mean Squared Error """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def fit(self, X, y, lr=1e-1, min_lr=1e-4, epoch=1000, verbose=False, plot_loss=False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        plot_x = []
        plot_y = []
        lr_step = (lr - min_lr) / epoch
        X = self.polynomial_preprocessing(X) if self.use_poly else np.array(X)
        y = np.array(y)
        n = len(y)

        self.m = np.random.randn(X.shape[1]) * 0.1

        for i in range(epoch):
            y_pred = self.predict(X)

            error = y - y_pred

            # Compute gradients
            dm = -(2 / n) * np.dot(X.T, error)
            db = -(2 / n) * np.sum(error)

            # Update parameters
            self.m -= lr * dm
            self.b -= lr * db

            if (i % max(1, (epoch // 100)) == 0) and plot_loss:
                plot_x.append(i)
                plot_y.append(self.rmse(y, y_pred))

            if verbose and (i % max(1, (epoch // 10)) == 0):
                print(f"Epoch {i} (lr {lr:.4f}): MSE={self.rmse(y, y_pred):.4f}")

            lr -= lr_step

        if plot_loss:
            plt.plot(plot_x, plot_y, label=f"RMSE")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.title("Loss during development")
            plt.legend()
            plt.show()

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
        X = self.polynomial_preprocessing(X) if self.use_poly else np.array(X)
        return np.dot(X, self.m) + self.b

    def measure_accuracy(self, test_X, test_y, tresh=0.5, plot_roc=False):
        # Predicted probabilities
        test_X = self.polynomial_preprocessing(
            test_X) if self.use_poly else np.array(test_X)
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
