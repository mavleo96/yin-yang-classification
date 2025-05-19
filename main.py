import os

import numpy as np

from src.config import DATA_CONFIG, OUTPUT_DIR
from src.data_generator import generate_yin_yang_data
from src.model import train_and_evaluate_models
from src.visualization import generate_visualizations


def main():
    np.random.seed(0)

    # Generate and split data
    X_train, X_test, y_train, y_test = generate_yin_yang_data(**DATA_CONFIG)

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Generate visualizations using all data points
    generate_visualizations(results, X_test, y_test)

    # Save model performance score to file
    with open(os.path.join(OUTPUT_DIR, "model_performance.txt"), "w") as f:
        text = ""
        for model_type, model_results in results.items():
            text += f"{model_type}\n"
            for model_result in model_results:
                text += f"{model_result['model']}: {model_result['accuracy']}\n"
            text += "\n"
        f.write(text)


if __name__ == "__main__":
    main()
