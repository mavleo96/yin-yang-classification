# Yin-Yang Classification

## Project Structure

```
.
├── .gitignore
├── INSTALL.md
├── LICENSE
├── README.md
├── requirements.txt
├── outputs/
├── main.py
└── src
    ├── __init__.py
    ├── config.py
    ├── data_generator.py
    ├── model.py
    └── visualization.py
```

## Features

- Generates synthetic Yin-Yang shaped data with customizable parameters
- Implements multiple classification models:
  - Random Forest
  - Multi-layer Perceptron (Neural Network)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
- Visualizes decision boundaries for each model
- Compares model performance

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd yin-yang-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to generate data, train models, and visualize results:
```bash
python main.py
```

Run below command to generate and save ground truth data: 
```bash
python -m src.data_generator
```

## Contributing

Feel free to:
- Add more models and modify configuration settings in config.py
- Implement additional visualization techniques
- Improve the documentation
- Add new evaluation metrics

## Changelog

#### 2025-05-19
- Added models: Naive Bayes, KMeans, DBSCAN

#### 2025-04-08
- Added configuration management through config.py
- Added models: MLP, KNN & XGBoost
- Created outputs directory for saving visualizations
- Added ground truth data generation script

#### 2021-08-28
- Project initialization
- Models: SVM, Logistic Regression & Random Forests

## License

This project is licensed under the MIT License - see the LICENSE file for details.
