# Credit Fraud Analyzer

## Overview
This Streamlit application analyzes credit card transactions data to detect fraudulent transactions. It includes features for data loading, exploration, feature analysis, data preprocessing, undersampling to address class imbalance, model training and evaluation, and data visualization.

## Features
- **Data Loading and Exploration:** Read a CSV file containing credit card transaction data into a pandas DataFrame. Generate histograms to visualize the class distribution in the data (fraudulent vs. non-fraudulent transactions). Check for and remove any duplicate rows in the DataFrame.
  
- **Feature Analysis:** Plot the distribution of transaction time and calculate the correlation of each feature with the target variable 'Class'. Create scatter plots for the features 'V11' and 'V17' against time for both fraudulent and non-fraudulent transactions.

- **Data Preprocessing:** Standardize the 'Amount' and 'Time' columns using the StandardScaler from sklearn.preprocessing. Drop the original 'Time' and 'Amount' columns from the DataFrame.

- **Undersampling:** Address the class imbalance problem using the RandomUnderSampler from the imblearn.under_sampling module to undersample the majority class (non-fraudulent transactions).

- **Model Training and Evaluation:** Split the undersampled data into a training set and a test set. Train a Logistic Regression model on the training data and make predictions on the test data. Calculate and print several evaluation metrics for the model, including accuracy, F1 score, precision, and recall. Print a classification report.

- **Data Visualization:** Generate various plots to visualize the distribution of the features and the target variable. Use seaborn and matplotlib libraries to create these plots. The plots include scatter plots comparing the amounts in fraud and normal transactions, scatter plots comparing 'Time' against 'Amount' for both normal and fraud transactions, and Kernel Density Estimation (KDE) plots for each feature in the dataset.

## Requirements
- Python 3.6+
- Streamlit
- Matplotlib
- Pandas
- Seaborn
- scikit-learn
- imbalanced-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/credit-fraud-analyzer.git
