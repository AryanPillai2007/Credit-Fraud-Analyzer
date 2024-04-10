# Credit Fraud Analyzer (CSV Analysis)
The primary goal of this project is to develop a comprehensive fraud detection system that enhances the security and trustworthiness of financial transactions.

Importing the required libraries is the first step of the script, including matplotlib, numpy, pandas, seaborn, imblearn, and sklearn components. 

It loads credit card transaction data from a CSV file (creditcard.csv) into a Pandas DataFrame (df). 
Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Next, the script searches the DataFrame for duplicate entries and eliminates them. In order to check for class imbalance in the target variable (Class), it also prints a summary of the data. df1 and df2, two copies of the original DataFrame, are made for various analytical and visualization uses.

    Exploratory Data Analysis (EDA)
To show the distribution of transaction time and amount in USD, the script creates two line graphs. The features with the highest positive (V11) and negative (V17) correlations with the target variable are determined by calculating the correlation matrix (corr). The relationship between time and these high-correlation indicators for both legitimate and fraudulent transactions is then visualized using dot scatter plots.

    Data preprocessing
The script produces new columns (amount and time) and uses StandardScaler to standardize the "Amount" and "Time" columns. The DataFrame's non-scaled columns are then removed (df2). After separating the target variable (Class) and features (X), an undersampling method (RandomUnderSampler) is used to correct for the target variable's class imbalance. The data is split into training and testing sets.

    Training and Evaluating the Model
Using the undersampled data, a logistic regression model is trained, and predictions are made using the test set. The script assesses the model using measures like accuracy, F1 score, precision, and recall and prints the percentage of fraud and non-fraud transactions in the dataset. Additionally included is a classification report that offers a thorough analysis of the model's performance.

    Visualization of Transaction Amount
A line plot is used by the script to display the distribution of transaction amounts for both legitimate and fraudulent transactions after loading the original dataset (df).

    Kernel Density Estimation (KDE) Plots
The script compares the distributions of normal and fraudulent transactions to create KDE plots for each feature. Time, amount, and other pertinent columns are among the characteristics. Plots such as this aid in pointing out trends and distinctions between the two classes.
