from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class YinYangClassifier:
    def __init__(self, model_type="random_forest", **kwargs):
        """
        Initialize the classifier

        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'mlp', 'svm')
        **kwargs : dict
            Additional parameters for the specific model
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on model_type"""
        model_map = {
            "random_forest": RandomForestClassifier,
            "mlp": MLPClassifier,
            "svm": SVC,
        }

        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model_map[self.model_type](**self.kwargs)

    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)

    def evaluate(self, X, y):
        """Evaluate the model on data"""
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0)
        return accuracy, report

    def get_feature_importance(self):
        """Get feature importance if available (for Random Forest)"""
        if self.model_type == "random_forest":
            return self.model.feature_importances_
        return None


def create_model_configs():
    """Create configurations for different models"""
    return [
        # Random Forest configurations with depths 1-10
        *[
            (
                "random_forest",
                {
                    "max_depth": depth,
                    "n_estimators": 50,  # Reduced from 100 to speed up training
                    "random_state": 42,
                    "n_jobs": -1,  # Use all available CPU cores
                },
            )
            for depth in range(1, 11)  # This will create depths 1 through 10
        ],
    ]


def create_model_name(model_type, params):
    """Create a descriptive name for the model based on its type and parameters"""
    if model_type == "random_forest":
        depth = params["max_depth"]
        return f"random_forest_depth_{depth}"
    elif model_type == "mlp":
        layers = params["hidden_layer_sizes"]
        return f"mlp_{len(layers)}_layers"
    else:  # svm
        kernel = params["kernel"]
        return f"svm_{kernel}"


def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple models with different configurations

    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels

    Returns:
    --------
    results : dict
        Dictionary containing results for each model
    """
    results = {}
    model_configs = create_model_configs()

    # Train and evaluate models with progress bar
    for model_type, params in tqdm(model_configs, desc="Training models"):
        # Create model name
        model_name = create_model_name(model_type, params)

        # Train and evaluate
        model = YinYangClassifier(model_type=model_type, **params)
        model.train(X, y)
        accuracy, report = model.evaluate(X, y)

        results[model_name] = {
            "accuracy": accuracy,
            "report": report,
            "model": model.model,
        }

    return results
