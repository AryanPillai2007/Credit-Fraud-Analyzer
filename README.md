# Credit Fraud Analyzer.

## Overview
This Streamlit application analyzes credit card transaction data to detect fraudulent activity. It has features for data loading, exploration, feature analysis, data preprocessing, undersampling to correct class imbalance, model training and evaluation, and data visualization.

## Features: 
- **Data Loading and Exploration** Read a CSV file containing credit card transaction data into a pandas DataFrame. Create histograms that show the class distribution in the data (fraudulent vs. non-fraudulent transactions). Check for and remove any duplicate entries from the DataFrame.
  
- **Feature analysis:** Plot the distribution of transaction time and compute the correlation between each attribute and the goal variable 'Class'. Create scatter plots of the features 'V11' and 'V17' against time for fraudulent and non-fraudulent transactions.

- **Data preprocessing:** Standardize the 'Amount' and 'Time' columns with sklearn.preprocessing's StandardScaler. Remove the original 'Time' and 'Amount' columns from the DataFrame.

- **Undersampling:** To address the class imbalance issue, use the RandomUnderSampler from the imblearn.under_sampling module to undersample the majority class (non-fraudulent transactions).

- **Model training and evaluation:** Separate the undersampled data into a training and test set. Train a Logistic Regression model on the training data and use it to predict the test data. Calculate and publish the model's assessment metrics, such as accuracy, F1 score, precision, and recall. Print a categorization report.

- Data Visualization: Create several plots to show the distribution of the features and the target variable. To make these plots, use the Seaborn and Matplotlib packages. The plots include scatter plots comparing amounts in fraud and normal transactions, scatter plots comparing 'Time' against 'Amount' for both normal and fraud transactions, and Kernel Density Estimation (KDE) plots for each dataset feature.

## Requirements
- Python 3.6+
- Streamlit
- Matplotlib
- Pandas
- Seaborn
- scikit-learn
- imbalanced-learn

## Thanks!
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


