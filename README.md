# Yin-Yang Classification

A fun project exploring different machine learning models' ability to classify data in a Yin-Yang shape pattern. This project implements and compares various classification models to understand their performance on this geometrically interesting dataset.


## Dataset Overview

The Yin-Yang dataset presents a visually complex, non-linear classification challenge with intertwined class regions. Its intricate structure challenges models to learn boundaries that are **non-linear, nested, and curved**—making it an excellent benchmark for comparing model expressiveness.

<img src="outputs/yin_yang_data.png" alt="Yin-Yang Data" width="300">

---

## Models and Results Visualization

### Random Forest Classifier
> _it ain't much but it's honest work_

- **Configuration**: 50 trees, varying `max_depth` from 1 to 9
- **Performance**: Captures most of the points in major classes at lower depths but is able to learn the complete distribution only at depth 9

<img src="outputs/random_forest.png" alt="Random Forest" width="600">

---
### XGBoost
> _This bad boy can fit so many f**king classes in it_


- **Configuration**: 50 estimators, varying `max_depth` from 1 to 3
- **Performance**: Shows solid performance even at low depths due to gradient boosting’s ability to combine weak learners.

<img src="outputs/xgboost.png" alt="XGBoost" width="600">

---

### Multi-Layer Perceptron (MLP)

#### MLP with Single Hidden Layer
- **Hidden Units**: 2 to 12
- **Performance**: Starts with baseline performance at low hidden units but improves with more neurons. Still, single-layer MLPs struggle to perfectly model the Yin-Yang’s nested, twisting structure.
  
<img src="outputs/mlp1.png" alt="MLP1" width="600">

#### MLP with Two Hidden Layers
- **Hidden Layers**: (2,2) to (12,12)
- **Performance**: Learns the boundaries with lower number of units per layer than MLP with single hidden layer. The second hidden layer allows the network to approximate more complex decision boundaries.

<img src="outputs/mlp2.png" alt="MLP2" width="600">

---

### Support Vector Machine (SVM)
- **Configuration**: SVM with `rbf`, `linear`, `poly` and `sigmoid` kernel and vary inverse regularization parameter between 0.1, 1 and 10.
- **Performance**:
  - **RBF kernel** captures the curved boundaries best.
  - **Linear and poly kernels** underperform due to their limited flexibility.
  - **Sigmoid kernel** gives unstable results in this context.
    > _Ew ... brother ew...what's that brother_
  - Very high training time for kernels like RBF.
  
<img src="outputs/svm.png" alt="SVM" width="600">

---

### K-Nearest Neighbors (KNN)
- **Neighbors**: 1 to 3
- **Performance**: Despite being simple, KNN performs surprisingly well on this dataset due to its instance-based nature. It handles the swirls of the Yin-Yang reasonably well.

<img src="outputs/knn.png" alt="KNN" width="600">

---

## Installation and Usage
See [INSTALL.md](INSTALL.md) for detailed installation and usage instructions.
