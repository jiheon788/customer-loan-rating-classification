# Customer Loan Rating Classification

## Project Overview

**Objective**: To develop a predictive model for classifying customer loan ratings based on an analysis of loan customer data.  
**Dataset Overview**: The dataset includes various features such as loan amount, loan duration, employment duration, home ownership status, annual income, and more.

## Execution Environment

The project requires the following software versions for execution:

```
Python 3.11.5
Jupyter Notebook 6.5.4
Anaconda 23.7.4
```

## Exploratory Data Analysis (EDA)

- Analysis of categorical and numerical variables.
- Exploration of the relationship between loan ratings and other variables.

## Data Preprocessing

- Handling Missing Values: Examination and treatment of missing values in the dataset.
- Categorical Variable Encoding: Application of One-Hot Encoding and Label Encoding to transform categorical variables.
- Numerical Variable Scaling: Utilization of Standard Scaling to scale numerical variables.

## Modeling

The F1 scores obtained after training and testing the following five models are as follows:

| Model               | F1 Score            |
| ------------------- | ------------------- |
| Random Forest       | 0.5567505123735182  |
| Logistic Regression | 0.427604644627017   |
| SVM                 | 0.37463334100814716 |
| Gradient Boosting   | 0.6562440051390214  |
| Neural Networks     | 0.760973963494476   |

Among these, the Neural Networks model exhibited the highest performance. The optimal parameters for this model were identified using GridSearchCV, as detailed below:

| Metric          | Value                            |
| --------------- | -------------------------------- |
| F1 Score        | 0.7766089561090138               |
| Best Parameters | - Activation: tanh               |
|                 | - Alpha: 0.0001                  |
|                 | - Hidden Layer Sizes: (100, 100) |
|                 | - Solver: sgd                    |
| Fits            | 108                              |
| Fold            | 3                                |

These results highlight the effectiveness of the Neural Networks model, particularly when optimized with the right set of hyperparameters.

### Data Augmentation

To address the issue of imbalanced datasets, data augmentation was performed using the SMOTE technique, which synthesizes new data points for minority classes.

As-Is:
![image](https://github.com/jiheon788/customer-loan-rating-classification/assets/90181028/28630d1e-cbff-4540-9340-98aaaf6134b9)

To-Be:
![image](https://github.com/jiheon788/customer-loan-rating-classification/assets/90181028/f893d23b-4d64-4a2a-997d-e66bea69573d)

After resolving the data imbalance issue, the optimal model identified earlier was trained.

```python
best_nn_model = MLPClassifier(
    activation='tanh',
    alpha=0.0001,
    hidden_layer_sizes=(100, 100),
    solver='sgd',
    max_iter=500,
    random_state=42
)

best_nn_model.fit(X_smote, y_smote)
```
