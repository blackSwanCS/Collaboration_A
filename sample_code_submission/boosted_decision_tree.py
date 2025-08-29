from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier
    """

    def __init__(self, name):
        self.model = XGBClassifier()
        self.scaler = StandardScaler()
        self.name = name

    def fit(self, train_data, labels, weights=None):
        X_train_data = self.scaler.fit_transform(train_data)
        self.model.fit(X_train_data, labels, sample_weight=weights)

    def predict(self, test_data):
        test_data = test_data.drop(columns=["score"], errors="ignore")
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
