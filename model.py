from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from config import MODEL_CONFIG


class YinYangClassifier:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._initialize_model()

    def _initialize_model(self):
        model_map = {
            "random_forest": RandomForestClassifier,
            "mlp": MLPClassifier,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "xgboost": XGBClassifier,
        }

        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model_map[self.model_type](**self.kwargs)

    def train(self, X, y):
        try:
            self.model.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Error training {self.model_type} model: {str(e)}")

    def evaluate(self, X, y):
        try:
            y_pred = self.model.predict(X)
            return accuracy_score(y, y_pred)
        except Exception as e:
            raise RuntimeError(f"Error evaluating {self.model_type} model: {str(e)}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    for model_config in tqdm(MODEL_CONFIG, desc="Training models"):
        try:
            model_type = model_config["model_type"]
            test_param = model_config["test_params"]
            kwargs = model_config["kwargs"]
            model = YinYangClassifier(model_type, **kwargs)
            model.train(X_train, y_train)
            if model_type not in results:
                results[model_type] = []
            results[model_type].append(
                {
                    "test_param_name": test_param,
                    "test_param_value": kwargs[test_param],
                    "accuracy": model.evaluate(X_test, y_test),
                    "model": model.model,
                }
            )
        except Exception as e:
            print(
                f"Error processing model {model_config.get('model_type', 'unknown')}: {str(e)}"
            )
            continue

    return results


if __name__ == "__main__":
    from config import DATA_CONFIG
    from data_generator import generate_yin_yang_data

    X_train, X_test, y_train, y_test = generate_yin_yang_data(**DATA_CONFIG)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
