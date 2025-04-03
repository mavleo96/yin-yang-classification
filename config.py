import numpy as np
from matplotlib.colors import ListedColormap

# Color configurations - High contrast color palette organized by class
COLORS = {
    "background": "#FFFFFF",  # Pure white background
    "class_colors": {
        -1: "#2B2D42",  # Dark blue-gray for misclassifications
        0: "#FF6B6B",  # Coral red for class 0
        1: "#4ECDC4",  # Turquoise for class 1
        2: "#FFD166",  # Bright yellow for class 2
        3: "#06D6A0",  # Emerald green for class 3
        4: "#118AB2",  # Ocean blue for class 4
    },
    "text": "#2B2D42",  # Dark blue-gray text color
    "grid": "#E0E0E0",  # Light gray grid color
}


# Create colormaps
def create_class_colormap(n_classes):
    """Create a colormap for classes including misclassification color"""
    colors = [COLORS["class_colors"][-1]]  # Start with misclassification color
    for i in range(n_classes):
        colors.append(COLORS["class_colors"][i])
    return ListedColormap(colors)


# Plot configurations
PLOT_CONFIG = {
    "figsize": (10, 8),
    "dpi": 300,
    "alpha": 0.6,
    "grid_alpha": 0.3,
    "fontsize": {"title": 16, "label": 14, "tick": 12},
    "font_family": "sans-serif",  # Modern sans-serif font
}

# Data generation configurations
DATA_CONFIG = {
    "n_samples": 200000,  # Doubled from 100000 for much higher density
    "radius": 500,
    "test_size": 0.2,
    "random_state": 42,
}
