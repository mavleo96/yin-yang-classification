import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from src.config import COLORS, OUTPUT_DIR, PLOT_CONFIG


def scatter_plot(X, y, title=None, filename=None, ax=None):
    sns.set_theme(
        style=PLOT_CONFIG["style"],
        font_scale=PLOT_CONFIG["font_scale"],
    )
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=COLORS["class_colors"],
        ax=ax,
        s=PLOT_CONFIG["s"],
        edgecolor=PLOT_CONFIG["edgecolor"],
    )
    ax.set_aspect(PLOT_CONFIG["aspect"], PLOT_CONFIG["adjust_aspect"])
    if title:
        ax.set_title(title)
    ax.axis(PLOT_CONFIG["axis"])
    ax.get_legend().remove()
    if filename:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(
            f"{OUTPUT_DIR}/{filename}",
            dpi=PLOT_CONFIG["dpi"],
            bbox_inches=PLOT_CONFIG["bbox_inches"],
        )
        plt.close()
    return ax


def generate_visualizations(results, X, y):
    for model_type, results in tqdm(results.items(), desc="Generating visualizations"):
        # Calculate the number of rows and columns for subplots
        n_plots = len(results)
        n_cols = min(3, n_plots) if n_plots != 4 else 2
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for i, result_item in enumerate(
            tqdm(results, desc=f"Generating visualizations for {model_type}")
        ):
            model = result_item["model"]
            test_params = result_item["test_params"]
            y_pred = model.predict(X)
            y_plot = np.where(y_pred == y, y, -1)

            ax = plt.subplot(n_rows, n_cols, i + 1)
            title = plot_title(model_type, test_params)
            scatter_plot(
                X,
                y_plot,
                title,
                ax=ax,
            )

        # Save the figure with all subplots
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(
            f"outputs/{model_type}.png",
            dpi=PLOT_CONFIG["dpi"],
            bbox_inches=PLOT_CONFIG["bbox_inches"],
        )
        plt.close()


def plot_title(model_type, param_dict):
    if model_type == "svm":
        mapping = {
            "linear": "Linear",
            "rbf": "RBF",
            "poly": "Polynomial",
            "sigmoid": "Sigmoid",
        }
        title = f"{mapping[param_dict['kernel']]} Kernel with C {param_dict['C']}"
        return title
    elif model_type == "mlp1":
        title = f"Single Layer Size {param_dict['hidden_layer_sizes'][0]}"
        return title
    elif model_type == "mlp2":
        title = f"Two Layers Size {param_dict['hidden_layer_sizes'][0]}"
        return title

    title = ""
    for key, value in param_dict.items():
        if isinstance(key, str):
            mapping = {
                "max_depth": "Tree Depth",
                "n_neighbors": "Number of Neighbors",
                "C": "C",
                "kernel": "Kernel",
                "hidden_layer_sizes": "Hidden Layer Sizes",
            }
            key = mapping[key] if key in mapping else key.replace("_", " ").title()
        elif isinstance(value, tuple):
            value = ", ".join(map(str, value))
        title += f"{key}: {value} "
    return title.strip()
