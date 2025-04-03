import numpy as np
import pandas as pd
import os
from config import PLOT_CONFIG, COLORS
from visualization import plot_data


def ensure_output_dir():
    """Ensure the outputs directory exists"""
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def generate_random_points(n_samples, radius):
    """Generate random points within a square of given radius"""
    x = np.random.uniform(-radius, radius, n_samples)
    y = np.random.uniform(-radius, radius, n_samples)
    return pd.DataFrame(zip(x, y), columns=["x", "y"])


def is_in_top_circle(row, radius):
    """Check if point is in the top small circle"""
    return row.x**2 + (row.y - radius / 2) ** 2 < (radius / 8) ** 2


def is_in_bottom_circle(row, radius):
    """Check if point is in the bottom small circle"""
    return row.x**2 + (row.y + radius / 2) ** 2 < (radius / 8) ** 2


def is_in_top_right_section(row, radius):
    """Check if point is in the top right section"""
    return (row.x > 0) & (row.x**2 + (row.y - radius / 2) ** 2 > (radius / 2) ** 2)


def is_in_bottom_left_section(row, radius):
    """Check if point is in the bottom left section"""
    return (row.x < 0) & (row.x**2 + (row.y + radius / 2) ** 2 < (radius / 2) ** 2)


def label_point(row, radius):
    """Label a single point based on its position"""
    if row.x**2 + row.y**2 > radius**2:
        return 0  # Outside the main circle
    elif is_in_top_circle(row, radius):
        return 1  # Small top circle
    elif is_in_bottom_circle(row, radius):
        return 2  # Small bottom circle
    elif is_in_top_right_section(row, radius) or is_in_bottom_left_section(row, radius):
        return 3  # Top right or bottom left section
    else:
        return 4  # Remaining sections


def generate_yin_yang_data(n_samples=20000, radius=500):
    """
    Generate Yin-Yang like dataset with multiple classes

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    radius : float
        Radius of the main circle

    Returns:
    --------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, 2)
    y : numpy.ndarray
        Target vector of shape (n_samples,)
    """
    # Generate random points
    data = generate_random_points(n_samples, radius)

    # Apply labels
    data["label"] = data.apply(label_point, axis=1, radius=radius)

    # Convert to numpy arrays
    X = data[["x", "y"]].values
    y = data["label"].values

    return X, y


def visualize_data(X, y, title="Yin-Yang Data", filename="yin_yang_data"):
    """
    Visualize the generated data using the consolidated visualization function

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    title : str
        Title for the plot
    filename : str
        Name of the file to save the plot
    """
    plot_data(X, y, title=title, filename=filename)


if __name__ == "__main__":
    # Generate and visualize sample data
    X, y = generate_yin_yang_data()
    visualize_data(X, y)
