import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def is_in_top_circle(row, radius):
    return row[0] ** 2 + (row[1] - radius / 2) ** 2 < (radius / 8) ** 2


def is_in_bottom_circle(row, radius):
    return row[0] ** 2 + (row[1] + radius / 2) ** 2 < (radius / 8) ** 2


def is_in_top_right_section(row, radius):
    return (row[0] > 0) & (row[0] ** 2 + (row[1] - radius / 2) ** 2 > (radius / 2) ** 2)


def is_in_bottom_left_section(row, radius):
    return (row[0] < 0) & (row[0] ** 2 + (row[1] + radius / 2) ** 2 < (radius / 2) ** 2)


def label_point(row, radius):
    if row[0] ** 2 + row[1] ** 2 > radius**2:
        return 0  # Outside the main circle
    elif is_in_top_circle(row, radius):
        return 1  # Small top circle
    elif is_in_bottom_circle(row, radius):
        return 2  # Small bottom circle
    elif is_in_top_right_section(row, radius) or is_in_bottom_left_section(row, radius):
        return 3  # Top right or bottom left section
    else:
        return 4  # Remaining sections


def generate_yin_yang_data(n_samples, radius, test_size=None, preprocess=False):
    # Generate random points
    X = np.random.uniform(-radius, radius, (n_samples, 2))
    y = np.apply_along_axis(label_point, 1, X, radius)

    # Preprocess data
    if preprocess:
        # Note: Normalization before splitting is not a good idea, but it's just for the sake of the example
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

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
    scatter_plot(X, y, title="Yin-Yang Data", filename="yin_yang_data.png")
