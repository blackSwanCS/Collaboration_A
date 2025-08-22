# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.isotonic import IsotonicRegression
# from sklearn.model_selection import train_test_split


# class BoostedDecisionTree:
#     """
#     XGBoost classifier + optional Isotonic Regression calibration
#     """

#     def __init__(self, name, use_calibration=False, calibration_split=0.2):
#         self.model = XGBClassifier()
#         self.scaler = StandardScaler()
#         self.name = name
#         self.use_calibration = use_calibration
#         self.calibration_split = calibration_split
#         self.calibrator = None  # isotonic regression object

#     def fit(self, train_data, labels, weights=None):
#         # (sub-train, calibration)
#         if self.use_calibration:
#             X_subtrain, X_calib, y_subtrain, y_calib, w_subtrain, w_calib = (
#                 train_test_split(
#                     train_data,
#                     labels,
#                     weights if weights is not None else [None] * len(labels),
#                     test_size=self.calibration_split,
#                     stratify=labels,
#                 )
#             )
#         else:
#             X_subtrain, y_subtrain, w_subtrain = train_data, labels, weights

#         X_subtrain_scaled = self.scaler.fit_transform(X_subtrain)

#         self.model.fit(X_subtrain_scaled, y_subtrain, sample_weight=w_subtrain)

#         if self.use_calibration:
#             X_calib_scaled = self.scaler.transform(X_calib)
#             raw_scores = self.model.predict_proba(X_calib_scaled)[:, 1]
#             self.calibrator = IsotonicRegression(out_of_bounds="clip")
#             self.calibrator.fit(raw_scores, y_calib)

#     def predict(self, test_data):
#         test_data = test_data.drop(columns=["score"], errors="ignore")
#         test_data = self.scaler.transform(test_data)
#         raw_scores = self.model.predict_proba(test_data)[:, 1]

#         if self.calibrator is not None:
#             return self.calibrator.transform(raw_scores)
#         return raw_scores

"""
BoostedDecisionTree - XGBoost + flexible calibration (isotonic / sigmoid / temperature)

Usage example:

bdt = BoostedDecisionTree(
    name="NP_plus",
    use_calibration=True,
    calibration_method="temperature",   # "isotonic" | "sigmoid" | "temperature"
    cv_calibration=False,               # True -> CalibratedClassifierCV with cv folds
    calibration_split=0.2,
)

bdt.fit(X_train, y_train, sample_weight=w_train)
p_cal = bdt.predict_proba(X_holdout)   # calibrated probabilities (if use_calibration=True)
p_raw = bdt.predict_raw(X_holdout)     # model's raw probabilities (pre-calibration)
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import check_array, check_X_y


class BoostedDecisionTree:
    def __init__(
        self,
        name: str = "BDT",
        use_calibration: bool = True,
        calibration_method: str = "isotonic",  # "isotonic" | "sigmoid" | "temperature"
        cv_calibration: bool = True,  # use CalibratedClassifierCV (cv folds) instead of single split
        calibration_split: float = 0.2,
        random_state: int = 42,
        xgb_params: Optional[dict] = None,
        early_stopping_rounds: int = 200,
        n_jobs: int = -1,
    ):
        self.name = name
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.cv_calibration = cv_calibration
        self.calibration_split = float(calibration_split)
        self.random_state = int(random_state)

        # calibration objects / params
        self.calibrator = None
        self.temperature_ = 1.0  # used when calibration_method == "temperature"

        # XGBoost default params tuned for smoother probabilities (you can override via xgb_params)
        default_xgb = dict(
            objective="binary:logistic",  # binary classification, outputs probabilities.
            eval_metric="logloss",
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            tree_method="hist",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=n_jobs,
        )
        if xgb_params is not None:
            default_xgb.update(xgb_params)
        self.xgb_params = default_xgb
        self.early_stopping_rounds = early_stopping_rounds

        self.model = XGBClassifier(**self.xgb_params)
        self.is_fitted = False
        self.best_iteration_ = None

    # ---------- Helpers ----------
    @staticmethod
    def _ensure_numpy(X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values
        return np.asarray(X)

    @staticmethod
    def _drop_score_column_if_present(X):
        if isinstance(X, pd.DataFrame) and "score" in X.columns:
            return X.drop(columns=["score"])
        return X

    # temperature scaling fit helper (simple grid search / NLL)
    """
    It searches over a grid of temperature values T > 0 and picks the T that minimizes negative log-likelihood (NLL) of the calibrated probabilities on the data. The chosen T is then used later to fix (calibrate) the model’s confidence without changing the ranking of examples.
    """

    @staticmethod
    def _fit_temperature_from_probs(
        p_raw: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit temperature T by minimizing NLL over a small grid.
        Returns best T (>0).
        """
        # ensure arrays
        p_raw = np.clip(np.asarray(p_raw), 1e-12, 1 - 1e-12)
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        logits = np.log(p_raw) - np.log(1.0 - p_raw)

        def nll_for_T(T):
            z = logits / T
            q = 1.0 / (1.0 + np.exp(-z))
            # negative log-likelihood
            eps = 1e-12
            if sample_weight is None:
                return -np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))
            else:
                w = sample_weight
                return -np.sum(
                    w * (y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))
                ) / (np.sum(w) + 1e-12)

        # grid search over plausible T values
        T_grid = np.concatenate(
            [
                np.linspace(0.5, 1.5, 41),
                np.linspace(1.5, 3.0, 31),
                np.linspace(0.1, 0.5, 20),
            ]
        )
        T_grid = np.unique(np.clip(T_grid, 0.01, 10.0))
        nlls = [nll_for_T(T) for T in T_grid]
        best_idx = int(np.argmin(nlls))
        return float(T_grid[best_idx])

    # ---------- Fit ----------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """
        Fit the XGBoost classifier and optional calibrator.

        Args:
            X: features (pandas DataFrame or numpy array)
            y: binary labels (0/1)
            sample_weight: optional per-sample weights (must be same length)
        """
        # Preprocess inputs
        if isinstance(X, pd.DataFrame):
            X_proc = self._drop_score_column_if_present(X)
        else:
            X_proc = X
        X_np = self._ensure_numpy(X_proc).astype("float32")
        y_np = np.asarray(y).astype("float32")
        sw = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype="float32")
        )

        # If using cross-validated calibration, we fit base model on whole training set first,
        # then CalibratedClassifierCV will perform internal CV to fit calibrator.
        # if I need calibration + cv + its isotnic or sigmoid
        if (
            self.use_calibration
            and self.cv_calibration
            and self.calibration_method in ("isotonic", "sigmoid")
        ):
            # fit base model on all data
            self.model.fit(X_np, y_np, sample_weight=sw, verbose=False)
            self.is_fitted = True

            # CalibratedClassifierCV expects an estimator that supports predict_proba
            self.calibrator = CalibratedClassifierCV(
                method=self.calibration_method, cv=5
            )
            # fit supports sample_weight argument (sklearn >= 0.24); pass weights if provided
            if sw is not None:
                self.calibrator.fit(X_np, y_np, sample_weight=sw)
            else:
                self.calibrator.fit(X_np, y_np)
            return

        # Otherwise, use a simple split: train model on subtrain, then calibrate on X_cal
        # if I need calibration
        if self.use_calibration:
            # stratify ensures similar nominal/shifted proportion in both sets
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X_np,
                y_np,
                test_size=self.calibration_split,
                stratify=y_np,
                random_state=self.random_state,
            )
            X_tr = X_tr.astype("float32")
            X_cal = X_cal.astype("float32")
            y_tr = y_tr.astype("float32")
            y_cal = y_cal.astype("float32")
            if sw is not None:
                sw_tr, sw_cal = train_test_split(
                    sw,
                    test_size=self.calibration_split,
                    stratify=y_np,
                    random_state=self.random_state,
                )
                sw_tr = sw_tr.astype("float32")
                sw_cal = sw_cal.astype("float32")
            else:
                sw_tr = sw_cal = None
        # if I dont need calibration
        else:
            X_tr, y_tr, sw_tr = (
                X_np.astype("float32"),
                y_np.astype("float32"),
                None if sw is None else sw.astype("float32"),
            )
            X_cal = y_cal = sw_cal = None

        # Fit base model with early stopping using X_cal as eval set (if calibration used).
        # if you need calibraton
        if self.use_calibration:
            eval_set = [(X_tr, y_tr), (X_cal, y_cal)]
            # XGBClassifier from sklearn API accepts sample_weight for fit
            self.model.fit(
                X_tr,
                y_tr,
                sample_weight=sw_tr,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
        else:
            # no calibrated eval set given -> still use early stopping with a small internal split
            self.model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)

        # if there is early stopping
        self.best_iteration_ = getattr(self.model, "best_iteration", None)
        self.is_fitted = True

        # If calibration requested, fit calibrator on X_cal / y_cal
        if self.use_calibration:
            # get raw model probabilities on calibration set
            raw_p = self.model.predict_proba(X_cal)[:, 1]

            if self.calibration_method in ("isotonic", "sigmoid"):
                # Use sklearn's CalibratedClassifierCV in 'prefit' mode to reuse self.model without refitting
                # Note: CalibratedClassifierCV(..., cv="prefit") is not allowed; we wrap differently:
                # We'll create a new CalibratedClassifierCV with cv="prefit" equivalent: use base estimator and cv="prefit"
                # sklearn API: CalibratedClassifierCV(base_estimator, method, cv="prefit") and then fit with calibration data
                calib = CalibratedClassifierCV(
                    self.model, method=self.calibration_method, cv="prefit"
                )
                # calib.fit expects X and y; it will use provided model's predict_proba on X
                if sw_cal is not None:
                    calib.fit(X_cal, y_cal, sample_weight=sw_cal)
                else:
                    calib.fit(X_cal, y_cal)
                self.calibrator = calib
            elif self.calibration_method == "temperature":
                # Fit temperature scaling on (raw_p, y_cal)
                T = self._fit_temperature_from_probs(raw_p, y_cal, sw_cal)
                self.temperature_ = float(T)
                self.calibrator = "temperature"  # marker
            else:
                raise ValueError(
                    f"Unknown calibration_method: {self.calibration_method}"
                )

    # ---------- Prediction ----------
    def predict_raw(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the model's raw probabilities (before calibration).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit(...) first.")
        if isinstance(X, pd.DataFrame):
            X_proc = self._drop_score_column_if_present(X)
        else:
            X_proc = X
        X_np = self._ensure_numpy(X_proc)
        # normal predict_proba
        p_raw = self.model.predict_proba(X_np)[:, 1]
        return np.clip(p_raw, 1e-12, 1 - 1e-12)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return calibrated probabilities if calibration used; otherwise returns raw probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit(...) first.")
        if isinstance(X, pd.DataFrame):
            X_proc = self._drop_score_column_if_present(X)
        else:
            X_proc = X
        X_np = self._ensure_numpy(X_proc)

        # 1) If calibrator is sklearn CalibratedClassifierCV (isotonic/sigmoid)
        if self.calibrator is not None and isinstance(
            self.calibrator, CalibratedClassifierCV
        ):
            p = self.calibrator.predict_proba(X_np)[:, 1]
            return np.clip(p, 1e-12, 1 - 1e-12)

        # 2) If temperature scaling
        if self.calibrator == "temperature" or (
            self.use_calibration and self.calibration_method == "temperature"
        ):
            p_raw = self.model.predict_proba(X_np)[:, 1]
            p_raw = np.clip(p_raw, 1e-12, 1 - 1e-12)
            logits = np.log(p_raw) - np.log(1 - p_raw)
            logits /= float(self.temperature_)
            p_temp = 1.0 / (1.0 + np.exp(-logits))
            return np.clip(p_temp, 1e-12, 1 - 1e-12)

        # 3) No calibrator -> return raw probabilities
        p_raw = self.model.predict_proba(X_np)[:, 1]
        return np.clip(p_raw, 1e-12, 1 - 1e-12)

    # def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    #     """
    #     Return binary predictions using calibrated probabilities and threshold.
    #     """
    #     p = self.predict_proba(X)
    #     return (p >= threshold).astype(int)

    # ---------- Utility ----------
    def get_params(self):
        return {
            "name": self.name,
            "use_calibration": self.use_calibration,
            "calibration_method": self.calibration_method,
            "cv_calibration": self.cv_calibration,
            "calibration_split": self.calibration_split,
            "xgb_params": self.xgb_params,
            "best_iteration": self.best_iteration_,
            "temperature": (
                self.temperature_
                if getattr(self, "temperature_", None) is not None
                else None
            ),
        }

    def __repr__(self):
        return f"<BoostedDecisionTree name={self.name} calibrated={self.use_calibration} method={self.calibration_method}>"
