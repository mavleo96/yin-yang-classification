import numpy as np
from sklearn.model_selection import train_test_split

RADIUS = 0.5


def is_in_top_circle(x, y):
    return x**2 + (y - RADIUS / 2) ** 2 < (RADIUS / 8) ** 2


def is_in_bottom_circle(x, y):
    return x**2 + (y + RADIUS / 2) ** 2 < (RADIUS / 8) ** 2


def is_in_top_right_section(x, y):
    return (x > 0) & (x**2 + (y - RADIUS / 2) ** 2 > (RADIUS / 2) ** 2)


def is_in_bottom_left_section(x, y):
    return (x < 0) & (x**2 + (y + RADIUS / 2) ** 2 < (RADIUS / 2) ** 2)


def label_point(row):
    x, y = row
    if x**2 + y**2 > RADIUS**2:
        return 0  # Outside the main circle
    elif is_in_top_circle(x, y):
        return 1  # Small top circle
    elif is_in_bottom_circle(x, y):
        return 2  # Small bottom circle
    elif is_in_top_right_section(x, y) or is_in_bottom_left_section(x, y):
        return 3  # Top right or bottom left section
    else:
        return 4  # Remaining sections


def generate_yin_yang_data(n_samples, test_size=None):
    # Yin Yang shaped data with circle radius of RADIUS
    # centered at the origin is generated

    # Generate random points
    X = np.random.uniform(-RADIUS, RADIUS, (n_samples, 2))
    y = np.apply_along_axis(label_point, 1, X)

    # Split into training and testing sets
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        return X_train, X_test, y_train, y_test
    else:
        return X, y


if __name__ == "__main__":
    import os
    import sys

    # Add the project root directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.config import DATA_CONFIG
    from src.visualization import scatter_plot

    # Generate and visualize sample data
    _, X, _, y = generate_yin_yang_data(**DATA_CONFIG)
    scatter_plot(X, y, filename="yin_yang_data.png")
