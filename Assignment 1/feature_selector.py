import os
import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product

class FeatureSelector():
    def __init__(self):
        pass

    def rmsle(self, y_true, y_pred):
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

    def select_features(
        self, 
        x_train, x_test, y_train, y_test, 
        save_every=10000, 
        start_err=float("inf"), 
        checkpoint_file="xgb_feature_combos.csv"
    ):
        # Convert data once
        X_train_np = x_train.to_numpy()
        X_test_np  = x_test.to_numpy()
        feature_names = x_train.columns.tolist()
        n_features = len(feature_names)

        if os.path.exists(checkpoint_file):
            df = pd.read_csv(checkpoint_file)
        else:
            # Generate all 0/1 combinations
            all_combos = list(product([0, 1], repeat=n_features))
            df = pd.DataFrame(all_combos, columns=feature_names)
            df["error"] = np.inf  # placeholder
            df.to_csv(checkpoint_file, index=False)

        # Parameters
        iteration_count = 0
        lowest_err = start_err
        best_combo = None

        # Load best from previous runs if available
        if (df["error"] != np.inf).any():
            best_idx = df["error"].idxmin()
            lowest_err = df.loc[best_idx, "error"]
            best_combo = df.loc[best_idx, feature_names].to_numpy().astype(bool)
            print(f"Resuming: best error so far = {lowest_err}")

        # Filter combos that still need evaluation
        to_evaluate = df[df["error"] == np.inf]

        for idx, row in to_evaluate.iterrows():
            mask = row[feature_names].to_numpy().astype(bool)
            
            # Skip if no features selected
            if not mask.any():
                df.at[idx, "error"] = np.inf
                continue

            # Prepare DMatrix for this combination
            dtrain = xgb.DMatrix(X_train_np[:, mask], label=y_train)
            dtest  = xgb.DMatrix(X_test_np[:, mask], label=y_test)

            booster = xgb.train({"objective": "reg:squarederror", "verbosity": 0}, dtrain, num_boost_round=100)
            y_pred = booster.predict(dtest)
            
            error = self.rmsle(y_test, y_pred)
            df.at[idx, "error"] = error

            # Update best combo
            if error < lowest_err:
                lowest_err = error
                best_combo = mask.copy()
                print("New lowest error:", error)

            iteration_count += 1

            # Save every `save_every` iterations
            if iteration_count % save_every == 0:
                df.to_csv(checkpoint_file, index=False)
                print(f"Saved checkpoint at iteration {iteration_count}")

        # Save final
        df.to_csv(checkpoint_file, index=False)

        if best_combo is None:
            raise RuntimeError("No valid feature combination was found or evaluated.")

        # Convert best mask to column names
        best_features = [f for f, m in zip(feature_names, best_combo) if m]
        return best_features
