# Magic-Telescope
## Overview
This project implements various machine learning models to classify high-energy gamma particles from hadron particles using the MAGIC Gamma Telescope dataset. The goal is to distinguish between signals produced by gamma rays (signal) and hadronic particles (background).

## Dataset
The dataset is from the UCI Machine Learning Repository:

Source: MAGIC Gamma Telescope Dataset

Features: 10 continuous attributes describing the shower properties

Target: Binary classification (gamma = 1, hadron = 0)

Samples: 19,020 instances

## Project Structure
magic-telescope/
├── data/
│   └── magic+gamma+telescope/  # Dataset directory
├── models/
│   └── best_model.h5           # Saved neural network model
├── magic_telescope_analysis.ipynb  # Main Jupyter notebook
├── requirements.txt            # Python dependencies
└── README.md                  # This file

## Key Features
### Data Preprocessing
Handling missing values (none found in this dataset)
Outlier detection and removal using IQR method
Feature standardization using StandardScaler
Class imbalance handling with RandomOversampler

## Models Implemented
Traditional ML Models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- ADA Boost
- Neural Network:

Customizable architecture with multiple hidden layers

Dropout regularization

Hyperparameter tuning for nodes, dropout, learning rate, and batch size

## Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Loss (for neural network)
Cross-validation scores

# Results
The best performing models were:

Random Forest (85.4% accuracy)
SVM (83.3% accuracy)
Gradient Boost (83.1% accuracy)

The neural network achieved competitive results after extensive hyperparameter tuning.

## Customization
To modify the analysis:

- Adjust hyperparameters in the model definitions
- Modify neural network architecture in the train_model() function
- Change evaluation metrics as needed
- Adjust visualization parameters for different plots

## Dependencies
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- tensorflow

## Citation
If you use this code or dataset, please cite:

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository

Original data generators: D. Heck et al., CORSIKA, A Monte Carlo code to simulate extensive air showers

## License
This project is open source and available under the MIT License.
