# SVM Implementation on Iris and MNIST (Digits) Datasets

## Overview

This project implements Support Vector Machines (SVM) to classify two popular datasets:
1. **Iris Dataset**: Classifies iris flowers into one of three species.
2. **MNIST (Digits) Dataset**: Classifies handwritten digits (0-9) from the `load_digits` dataset in `scikit-learn`.

---

## Datasets

### 1. Iris Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)
- **Features**: 
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Classes**: 
  - Setosa
  - Versicolor
  - Virginica

### 2. MNIST (Digits) Dataset
- **Source**: `scikit-learn` (`load_digits`)
- **Features**: 8x8 pixel images of handwritten digits (flattened into 64 features)
- **Classes**: Digits from 0 to 9

---

## Requirements

Install the necessary Python libraries using the following command:

```bash
pip install numpy matplotlib scikit-learn
