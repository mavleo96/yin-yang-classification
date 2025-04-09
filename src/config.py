# Output directory
OUTPUT_DIR = "outputs"


# Data generation configurations
DATA_CONFIG = {
    "n_samples": 50000,
    "radius": 500,
    "test_size": 0.5,
}

# Plot configurations
PLOT_CONFIG = {
    "style": "white",
    "font_scale": 2,
    "s": 20,
    "edgecolor": "none",
    "aspect": "equal",
    "adjust_aspect": "box",
    "axis": "off",
    "dpi": 300,
    "bbox_inches": "tight",
}


# Color configurations - High contrast color palette organized by class
COLORS = {
    "class_colors": {
        -1: "#2B2D42",  # Dark blue-gray for misclassifications
        0: "#FF6B6B",  # Coral red for class 0
        1: "#4ECDC4",  # Turquoise for class 1
        2: "#FFD166",  # Bright yellow for class 2
        3: "#06D6A0",  # Emerald green for class 3
        4: "#118AB2",  # Ocean blue for class 4
    },
}


# Model configurations
MODEL_CONFIG = [
    *[
        {
            "model_type": "random_forest",
            "test_params": {
                "max_depth": depth,
            },
            "kwargs": {
                "n_estimators": 50,
                "random_state": 0,
            },
        }
        for depth in range(1, 10)
    ],
    *[
        {
            "model_type": "mlp1",
            "test_params": {
                "hidden_layer_sizes": (size,),
            },
            "kwargs": {
                "max_iter": 1000,
                "random_state": 0,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
        }
        for size in range(2, 13, 2)
    ],
    *[
        {
            "model_type": "mlp2",
            "test_params": {
                "hidden_layer_sizes": (size, size),
            },
            "kwargs": {
                "max_iter": 1000,
                "random_state": 0,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
        }
        for size in range(2, 13, 2)
    ],
    *[
        {
            "model_type": "svm",
            "test_params": {
                "kernel": kernel,
                "C": C,
            },
            "kwargs": {
                "random_state": 0,
                "probability": True,
            },
        }
        for kernel in ["rbf", "linear", "poly", "sigmoid"]
        for C in [0.1, 1, 10]
    ],
    *[
        {
            "model_type": "knn",
            "test_params": {
                "n_neighbors": n_neighbors,
            },
            "kwargs": {
                "weights": "uniform",
                "algorithm": "auto",
            },
        }
        for n_neighbors in range(1, 4)
    ],
    *[
        {
            "model_type": "xgboost",
            "test_params": {
                "max_depth": max_depth,
            },
            "kwargs": {
                "n_estimators": 50,
                "random_state": 0,
            },
        }
        for max_depth in range(1, 4)
    ],
]
