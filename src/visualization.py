import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from .config import COLORS, OUTPUT_DIR, PLOT_CONFIG


def scatter_plot(X, y, title, filename=None, ax=None):
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


# def plot_model_comparison(results, filename=None):

#     # Extract model names and accuracies
#     model_names = list(results.keys())
#     accuracies = [results[name]["accuracy"] for name in model_names]

#     # Create bar plot
#     plt.style.use("default")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_facecolor(COLORS["background"])
#     fig.patch.set_facecolor(COLORS["background"])

#     # Create bar chart with modern colors
#     bars = ax.bar(model_names, accuracies, color=COLORS["class_colors"][0], alpha=0.8)

#     # Add value labels on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(
#             bar.get_x() + bar.get_width() / 2.0,
#             height + 0.01,
#             f"{height:.3f}",
#             ha="center",
#             va="bottom",
#             color=COLORS["text"],
#         )

#     # Style the plot
#     ax.set_title("Model Accuracy Comparison", fontsize=14, pad=20, color=COLORS["text"])
#     ax.set_ylabel("Accuracy", fontsize=12, color=COLORS["text"])
#     ax.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1
#     ax.grid(axis="y", linestyle="--", alpha=0.3, color=COLORS["grid"])

#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45, ha="right", color=COLORS["text"])
#     plt.yticks(color=COLORS["text"])

#     # Adjust layout and save if filename provided
#     plt.tight_layout()
#     if filename:
#         plt.savefig(
#             f"{OUTPUT_DIR}/{filename}.png",
#             dpi=300,
#             bbox_inches="tight",
#             facecolor=COLORS["background"],
#             edgecolor="none",
#             pad_inches=0.1,
#         )
#     plt.close()
