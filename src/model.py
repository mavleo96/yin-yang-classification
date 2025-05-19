import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from src.config import MODEL_CONFIG


class YinYangClassifier:
    def __init__(self, model_type, task, **kwargs):
        self.model_type = model_type
        self.task = task
        self.kwargs = kwargs
        self.model = self._initialize_model()

    def _initialize_model(self):
        model_map = {
            "random_forest": RandomForestClassifier,
            "mlp1": MLPClassifier,
            "mlp2": MLPClassifier,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "xgboost": XGBClassifier,
            "naive_bayes": GaussianNB,
            "kmeans": KMeans,
            "dbscan": DBSCAN,
        }

        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model_map[self.model_type](**self.kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.predict(X, y)
        return accuracy_score(y, y_pred)

    def predict(self, X, y):
        if hasattr(self.model, "predict"):
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model.fit_predict(X)
        if self.task == "clustering":
            return self._label_assignment(y, y_pred)
        else:
            return y_pred

    @staticmethod
    def _label_assignment(y, y_pred):
        # The labels from the clustering model will not align with labels from the dataset
        # We use the Hungarian algorithm to find the best mapping between the two sets of labels
        # Noisy labels i.e -1 (like in case of DBSCAN) are excluded but retained as -1

        labels = np.unique(np.concatenate((y, y_pred)))
        labels = labels[labels != -1]

        cm = confusion_matrix(y, y_pred, labels=labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {labels[col]: labels[row] for row, col in zip(row_ind, col_ind)}

        y_pred_mapped = np.array([mapping.get(label, -1) for label in y_pred])

        return y_pred_mapped


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    for model_config in tqdm(MODEL_CONFIG, desc="Training models"):
        try:
            model_type = model_config["model_type"]
            task = model_config["task"]
            test_params = model_config["test_params"]
            kwargs = model_config["kwargs"]
            model = YinYangClassifier(
                model_type,
                task,
                **test_params,
                **kwargs,
            )
            model.train(X_train, y_train)
            if model_type not in results:
                results[model_type] = []
            results[model_type].append(
                {
                    "test_params": test_params,
                    "accuracy": model.evaluate(X_test, y_test),
                    "model": model,
                }
            )
        except Exception as e:
            print(
                f"Error processing model {model_config.get('model_type', 'unknown')}: {str(e)}"
            )
            continue

    return results


if __name__ == "__main__":
    import os
    import sys

    # Add the project root directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.config import DATA_CONFIG
    from src.data_generator import generate_yin_yang_data

    X_train, X_test, y_train, y_test = generate_yin_yang_data(**DATA_CONFIG)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
