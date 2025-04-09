import numpy as np

from src.config import DATA_CONFIG
from src.data_generator import generate_yin_yang_data
from src.model import train_and_evaluate_models
from src.visualization import generate_visualizations


def main():
    np.random.seed(0)

    # Generate and preprocess data
    X_train, X_test, y_train, y_test = generate_yin_yang_data(
        **DATA_CONFIG, preprocess=True
    )

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Generate visualizations using all data points
    generate_visualizations(results, X_test, y_test)


if __name__ == "__main__":
    main()
