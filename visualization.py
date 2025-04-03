import numpy as np
import matplotlib.pyplot as plt
import os
from config import PLOT_CONFIG, COLORS, create_class_colormap
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm


def ensure_output_dir():
    """Ensure the outputs directory exists"""
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def style_plot(ax, title):
    """Apply consistent styling to plots"""
    # Set modern font
    plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]

    ax.set_title(
        title, fontsize=PLOT_CONFIG["fontsize"]["title"], color=COLORS["text"], pad=20
    )
    ax.set_xlabel("X", fontsize=PLOT_CONFIG["fontsize"]["label"], color=COLORS["text"])
    ax.set_ylabel("Y", fontsize=PLOT_CONFIG["fontsize"]["label"], color=COLORS["text"])
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], color=COLORS["grid"])
    ax.tick_params(
        axis="both", which="major", labelsize=PLOT_CONFIG["fontsize"]["tick"]
    )


def plot_data(X, y, title="Data Visualization", filename=None):
    """
    Plot data points

    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    y : numpy.ndarray
        Target labels
    title : str
        Plot title
    filename : str, optional
        Filename to save the plot
    """
    ensure_output_dir()

    # Create plot with pure white background
    plt.style.use("default")
    plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    # Get number of unique classes
    n_classes = len(np.unique(y))

    # Create colormap for classes
    cmap = create_class_colormap(n_classes)

    # Plot points with their class colors
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.7, s=15)

    # Style the plot
    ax.set_title(title, fontsize=14, pad=20, color=COLORS["text"])

    # Remove axes for cleaner look
    ax.set_axis_off()

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Adjust layout and save if filename provided
    plt.tight_layout()
    if filename:
        plt.savefig(
            f"outputs/{filename}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=COLORS["background"],
            edgecolor="none",
            pad_inches=0.1,
            transparent=False,
        )
    plt.close()


def plot_classification_results(
    X, y, model, title="Classification Results", filename=None
):
    """
    Plot classification results showing correct and incorrect predictions

    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    y : numpy.ndarray
        True labels
    model : object
        Trained model
    title : str
        Plot title
    filename : str, optional
        Filename to save the plot
    """
    ensure_output_dir()

    # Create plot with pure white background
    plt.style.use("default")
    plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    # Get predictions
    y_pred = model.predict(X)

    # Create point colors based on correct/incorrect classification
    point_colors = []
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            # Correct classification - use the class color
            point_colors.append(COLORS["class_colors"][y[i]])
        else:
            # Misclassification - use the misclassification color
            point_colors.append(COLORS["class_colors"][-1])

    # Plot points with colors based on classification
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, alpha=0.7, s=15)

    # Add legend
    legend_elements = []
    unique_classes = np.unique(y)
    for i in unique_classes:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["class_colors"][i],
                markersize=10,
                label=f"Class {i} (Correct)",
            )
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["class_colors"][-1],
            markersize=10,
            label="Misclassified",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor=COLORS["background"],
        edgecolor=COLORS["grid"],
        fontsize=10,
    )

    # Style the plot
    ax.set_title(title, fontsize=14, pad=20, color=COLORS["text"])

    # Remove axes for cleaner look
    ax.set_axis_off()

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Adjust layout and save if filename provided
    plt.tight_layout()
    if filename:
        plt.savefig(
            f"outputs/{filename}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=COLORS["background"],
            edgecolor="none",
            pad_inches=0.1,
            transparent=False,
        )
    plt.close()


def plot_model_comparison(results, filename=None):
    """Plot comparison of model accuracies"""
    ensure_output_dir()

    # Extract model names and accuracies
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    # Create bar plot
    plt.style.use("default")
    plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    # Create bar chart with modern colors
    bars = ax.bar(model_names, accuracies, color=COLORS["class_colors"][0], alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            color=COLORS["text"],
        )

    # Style the plot
    ax.set_title("Model Accuracy Comparison", fontsize=14, pad=20, color=COLORS["text"])
    ax.set_ylabel("Accuracy", fontsize=12, color=COLORS["text"])
    ax.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=COLORS["grid"])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", color=COLORS["text"])
    plt.yticks(color=COLORS["text"])

    # Adjust layout and save if filename provided
    plt.tight_layout()
    if filename:
        plt.savefig(
            f"outputs/{filename}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=COLORS["background"],
            edgecolor="none",
            pad_inches=0.1,
            transparent=False,
        )
    plt.close()
