# Yin-Yang Classification

A fun project exploring different machine learning models' ability to classify data in a Yin-Yang shape pattern. This project compares various models including Random Forests of different depths, Multi-layer Perceptrons, and Support Vector Machines.

## Project Structure

```
yin-yang-classification/
├── data_generator.py    # Generates Yin-Yang shaped data
├── models.py           # Implements different classifiers
├── visualization.py    # Visualization utilities
├── main.py            # Main script to run the project
└── requirements.txt    # Project dependencies
```

## Features

- Generates synthetic Yin-Yang shaped data
- Implements multiple classification models:
  - Random Forest (with varying depths)
  - Multi-layer Perceptron (Neural Network)
  - Support Vector Machine
- Visualizes decision boundaries
- Compares model performance
- Detailed classification reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yin-yang-classification.git
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

This will:
1. Generate Yin-Yang shaped data
2. Train multiple models with different configurations
3. Display accuracy scores and classification reports
4. Show decision boundaries for selected models
5. Plot a comparison of model performances

## Results

The project will output:
- Visualizations of the generated data
- Accuracy scores for each model
- Detailed classification reports
- Decision boundary plotsx
- Model comparison bar chart

## Contributing

Feel free to:
- Add more models
- Experiment with different data generation parameters
- Implement additional visualization techniques
- Improve the documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
