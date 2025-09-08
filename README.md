# Titanic-Survival-Prediction
# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Key Findings](#key-findings)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning techniques. The analysis employs three different classification algorithms: Logistic Regression, Decision Tree, and Support Vector Machine (SVM) to compare their performance and identify the most important factors affecting survival.

## Dataset

The dataset contains information about Titanic passengers with the following original features:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes) - Target variable
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Features

After data preprocessing, the following features are used for modeling:
- **Pclass**: Passenger class (numerical)
- **Sex**: Gender (encoded: male=1, female=0)
- **Age**: Age (missing values filled with median)
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Embarked_S**: Embarked from Southampton (binary)
- **Embarked_C**: Embarked from Cherbourg (binary)
- **Embarked_Q**: Embarked from Queenstown (binary)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

## Usage

### Running the Jupyter Notebook
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `titanic.ipynb`
3. Run all cells sequentially

### Running as Python Script
```python
python titanic_prediction.py
```

### Input Data
Ensure you have the `titanic.csv` file in the same directory as the notebook.

## Models

### 1. Logistic Regression
- **Accuracy**: 80%
- **Precision**: 82% (Class 0), 77% (Class 1)
- **Recall**: 85% (Class 0), 74% (Class 1)
- **F1-Score**: 84% (Class 0), 76% (Class 1)

### 2. Decision Tree Classifier
- **Accuracy**: 78%
- **Precision**: 79% (Class 0), 76% (Class 1)
- **Recall**: 85% (Class 0), 69% (Class 1)
- **F1-Score**: 82% (Class 0), 72% (Class 1)

### 3. Support Vector Machine (Linear Kernel)
- **Accuracy**: 78%
- **Precision**: 80% (Class 0), 75% (Class 1)
- **Recall**: 84% (Class 0), 70% (Class 1)
- **F1-Score**: 82% (Class 0), 73% (Class 1)

## Key Findings

### Logistic Regression Insights
The logistic regression model reveals the following coefficient impacts:
- **Sex (-1.28)**: Being male significantly decreases survival chances
- **Pclass (-0.86)**: Lower passenger class reduces survival probability
- **Age (-0.40)**: Older passengers had lower survival rates
- **SibSp (-0.31)**: Having more siblings/spouses aboard decreased survival
- **Embarked_S (-0.27)**: Embarking from Southampton had negative impact

### Decision Tree Feature Importance
The decision tree identifies feature importance as:
- **Sex (0.36)**: Most important factor
- **Age (0.32)**: Second most important
- **Pclass (0.13)**: Third most important
- **SibSp (0.09)**: Moderate importance
- **Other features**: Minimal impact

### SVM Analysis
The SVM model primarily relies on:
- **Sex (-0.96)**: Overwhelming factor in predictions
- **Other features**: Minimal coefficients, indicating less influence

### Survival Factors Summary

**Positive Impact on Survival:**
- Being female
- Younger age
- Traveling in higher-class cabins (1st class)
- Having fewer family members aboard
- Embarking from Cherbourg or Queenstown

**Negative Impact on Survival:**
- Being male
- Older age
- Traveling in lower-class cabins (3rd class)
- Having more siblings/spouses or parents/children aboard
- Embarking from Southampton

## File Structure
```
titanic-survival-prediction/
│
├── titanic.ipynb          # Main Jupyter notebook
├── titanic.csv            # Dataset file
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Data Preprocessing Steps

1. **Feature Selection**: Removed irrelevant features (PassengerId, Name, Ticket, Fare, Cabin)
2. **Encoding**: Converted categorical variables to numerical (Sex, Embarked)
3. **Missing Values**: 
   - Filled missing Age values with median
   - Filled missing Embarked values with 0
4. **One-Hot Encoding**: Applied to Embarked column
5. **Standardization**: Applied StandardScaler to all features
6. **Train-Test Split**: 80-20 split with random_state=42

## Model Performance Comparison

| Model | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|-------|----------|----------------|--------------|----------------|
| Logistic Regression | 80% | 80% | 80% | 80% |
| Decision Tree | 78% | 78% | 77% | 77% |
| SVM (Linear) | 78% | 78% | 77% | 77% |

**Best Performing Model**: Logistic Regression with 80% accuracy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements

- Feature engineering (creating new features like family size, title extraction from names)
- Hyperparameter tuning using GridSearchCV
- Ensemble methods (Random Forest, Gradient Boosting)
- Cross-validation for more robust evaluation
- ROC-AUC analysis
- Handling class imbalance with techniques like SMOTE

## Acknowledgments

- Kaggle for providing the Titanic dataset
- The machine learning community for valuable insights and techniques
- Scikit-learn documentation and examples

