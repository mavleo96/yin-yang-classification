from data_generator import generate_yin_yang_data, visualize_data
from models import train_and_evaluate_models
from visualization import plot_data, plot_model_comparison, plot_classification_results
from sklearn.preprocessing import StandardScaler
from config import DATA_CONFIG
from tqdm import tqdm
import numpy as np


def generate_and_preprocess_data():
    """Generate and preprocess the Yin-Yang dataset"""
    print("Generating Yin-Yang data...")
    with tqdm(total=100, desc="Data generation") as pbar:
        X, y = generate_yin_yang_data()
        pbar.update(50)
        visualize_data(X, y, "Generated Yin-Yang Data", "yin_yang_data")
        pbar.update(50)
    print("✓ Saved data visualization to outputs/yin_yang_data.png")

    print("\nPreprocessing data...")
    with tqdm(total=100, desc="Data preprocessing") as pbar:
        # No train-test split, use all data
        pbar.update(50)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pbar.update(50)

    return X, y


def print_results(results):
    """Print model evaluation results"""
    print("\nModel Results:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print("Classification Report:")
        print(result["report"])


def generate_visualizations(results, X, y):
    """Generate and save all visualizations"""
    print("\nGenerating visualizations...")
    with tqdm(total=100, desc="Creating plots") as pbar:
        # Plot model comparison
        plot_model_comparison(results, "model_comparison")
        pbar.update(20)
        print("✓ Saved model comparison to outputs/model_comparison.png")

        # Plot classification results for each model
        for model_name, result in results.items():
            model = result["model"]
            plot_classification_results(
                X,
                y,
                model,
                title=f"Classification Results - {model_name}",
                filename=f"classification_results_{model_name}",
            )
            print(
                f"✓ Saved classification results for {model_name} to outputs/classification_results_{model_name}.png"
            )
            pbar.update(80 / len(results))

    print("\nAll visualizations have been saved to the outputs/ directory.")


def main():
    """Main execution function"""
    # Generate and preprocess data
    X, y = generate_and_preprocess_data()

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X, y)

    # Print results
    print_results(results)

    # Generate visualizations using all data points
    generate_visualizations(results, X, y)


if __name__ == "__main__":
    main()
