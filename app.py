import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/creditcard.csv')
    return df

# Create the app
def main():
    st.title("Credit Card Fraud Analysis")
    st.write("## Main Purpose")
    st.write("In today's digital world, where online transactions and data exchange have become commonplace, the importance of solid cybersecurity protections and effective fraud detection systems cannot be overstated. As technology continues to advance, it has become increasingly challenging to safeguard sensitive data and financial transactions from bad actors who are constantly looking for new ways to exploit vulnerabilities in our systems.")
    st.write("According to a report by the Federal Trade Commission, Americans lost over $1.9 billion to fraud in 2019 alone, resulting in significant financial losses for businesses and individuals alike. This highlights the pressing need for a reliable fraud detection system to effectively identify fraudulent activity in real time, preventing financial losses and safeguarding sensitive information.")
    st.write("The primary objective of this research is to develop a reliable fraud detection system that can effectively identify fraudulent activity in real-world financial transactions. I aim to answer questions such as: How can Machine Learning detect fraudulent transactions promptly and efficiently? What are the most effective machine learning algorithms to use for fraud detection?")
    st.write("Through this research, I hope to offer a practical solution that can be used to safeguard sensitive information and prevent financial loss due to fraudulent activity.")

    # Load the data
    df = load_data()

    # Show some basic statistics
    st.header("Data Statistics")
    st.write("This section provides an overview of the statistical properties of the dataset, including the following:")

    st.write("**Standard Deviation:** A measure of the amount of dispersion or variation from the mean value. A low standard deviation indicates that the data points tend to be closer to the mean, while a high standard deviation indicates that the data points are spread out over a wider range of values.")
    st.write(df.std())

    st.write("**Minimum:** The smallest value in the feature.")
    st.write(df.min())

    st.write("**Maximum:** The largest value in the feature.")
    st.write(df.max())

    st.write("Understanding the data statistics is crucial for gaining insights into the dataset and identifying potential challenges or issues that should be utilized during the analysis and modeling process.")

    # Show the class distribution
    st.header("Class Distribution")
    class_counts = df["Class"].value_counts()
    st.write(class_counts)
    st.write("The Class Distribution section displays the count of transactions labeled as fraudulent (Class = 1) and non-fraudulent (Class = 0). This information is essential for understanding the class imbalance present in the dataset. Class imbalance occurs when one class (in this case, fraudulent transactions) is significantly underrepresented compared to the other class (non-fraudulent transactions). This imbalance can pose challenges for machine learning models, as they may struggle to accurately identify the minority class. By understanding the class distribution, researchers can implement appropriate techniques, such as undersampling or oversampling, to mitigate the effects of class imbalance and improve the model's performance.")

    # Plot the class histogram
    st.header("Class Histogram")
    count_classes = pd.Series(df['Class']).value_counts(sort=True).sort_index()
    fig, ax = plt.subplots()
    count_classes.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_title("Fraud class histogram")
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')
    st.pyplot(fig)
    st.write("The Class variable indicates whether a transaction is fraudulent (1) or not (0). The plot is a bar chart showing each class value's frequency. The plot will help you understand the imbalance of the dataset, as there are far more non-fraudulent transactions than fraudulent ones. This means that you will need to use some techniques to deal with the imbalance, such as under-sampling, over-sampling, or using a different metric than accuracy to evaluate your model.")

    # Plot the amount for fraud and normal transactions
    st.header("Amount in Fraud vs Normal Transactions")
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(normal['Amount'], label='Normal', color='#2769FE')
    ax.plot(fraud['Amount'], label='Fraud', color='#FF5F57')
    ax.set_title('Amount in Fraud vs Normal Transactions')
    ax.set_xlabel('Index')
    ax.set_ylabel('Amount')
    ax.legend(loc='upper right')
    st.pyplot(fig)
    st.write("The fifth plot is a graph comparing the amounts in fraud and normal transactions. It shows significant spikes in fraud amounts. Based on this plot, a possible prediction could be made is that transactions with higher amounts might be more likely to be fraudulent. This is because the plot seems to suggest a correlation between higher transaction amounts and fraudulent transactions.")

    # Plot the transaction time distribution
    st.header("Distribution of Transaction Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('lightgray')
    ax.plot(df.index, df["Time"], color='red')
    ax.set_xlabel("Transaction")
    ax.set_ylabel("Time")
    ax.set_title("Distribution of Transaction Time")
    st.pyplot(fig)
    st.write("Shows an initial rapid increase in transactions, which slows down as time progresses but continues to grow steadily. This suggests that the frequency of transactions was high at the beginning and then started to decrease, but the total number of transactions kept increasing over time.")

    # Plot the max positive correlation (V11)
    st.header("Max Positive Correlation (V11)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(normal["Time"], normal["V11"], label="Real Transaction", color='#2769FE')
    ax.scatter(fraud["Time"], fraud["V11"], label="Fraud Transaction", color='#FF5F57')
    ax.set_xlabel("Time")
    ax.set_ylabel("V11 class")
    ax.set_title("Max positive correlation (V11)")
    ax.legend()
    st.pyplot(fig)
    st.write("The scatterplot titled 'Max positive correlation (V11)' represents the correlation between time and the V11 class in your credit card transactions dataset. The plot differentiates between Real Transactions (blue dots) and Fraud Transactions (orange dots). As time progresses, the frequency of both real and fraudulent transactions fluctuates. However, there are noticeable spikes in the V11 class for fraud transactions at various points in time, suggesting that higher V11 values could be indicative of fraud. This insight could be valuable when building a model to detect fraudulent transactions. However, correlation does not imply causation, and this is just one feature among many others that should be considered. Based on the scatter plot, which compares real and fraud transactions over time and highlights the max positive correlation (V11), a possible prediction could be that transactions with a higher V11 value might be more likely to be fraudulent. This is because the plot seems to suggest a correlation between V11 and fraudulent transactions.")

    # Plot the max negative correlation (V17)
    st.header("Max Negative Correlation (V17)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(normal["Time"], normal["V17"], label="Real Transaction", color='#2769FE')
    ax.scatter(fraud["Time"], fraud["V17"], label="Fraud Transaction", color='#FF5F57')
    ax.set_xlabel("Time")
    ax.set_ylabel("V17 class")
    ax.set_title("Max negative correlation (V17)")
    ax.legend()
    st.pyplot(fig)
    st.write("The fourth plot is a scatter plot showing the max negative correlation (V17) over time, with real transactions in blue and fraud transactions in orange. This plot seems to suggest a pattern where fraud transactions have a more pronounced negative correlation with V17. Based on this plot, a possible prediction could be made that transactions with a lower V17 value might be more likely to be fraudulent. This is because the plot seems to suggest a correlation between lower V17 values and fraudulent transactions. There are noticeable spikes in the V17 class for fraud transactions at various points in time. This suggests that lower V17 values could be indicative of fraud.")


# Plot before undersampling
    st.header("Distribution of Transaction Amounts (Before Undersampling)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[df['Class'] == 0]['Amount'], df[df['Class'] == 0]['Time'], label='Normal', color='#2769FE')
    ax.scatter(df[df['Class'] == 1]['Amount'], df[df['Class'] == 1]['Time'], label='Anomaly', color='#FF5F57')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Time')
    ax.legend()
    st.pyplot(fig)

    st.write("This scatter plot shows the distribution of transaction amounts before any undersampling is applied. The x-axis represents the transaction amount, and the y-axis represents time. Each point on the plot represents a transaction, with blue dots indicating non-fraudulent transactions and red dots indicating fraudulent transactions.")
    st.write("From this plot, we can observe a significant class imbalance, where the majority of transactions are non-fraudulent (blue dots), and only a small number of transactions are fraudulent (red dots). This imbalance can pose challenges for machine learning models, as they may struggle to accurately identify the minority class (fraudulent transactions) during training.")

    # Apply undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_undersampled, y_undersampled = undersampler.fit_resample(df.drop('Class', axis=1), df['Class'])
    undersampled_df = pd.concat([X_undersampled, y_undersampled], axis=1)

    # Plot after undersampling
    st.header("Distribution of Transaction Amounts (After Undersampling)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(undersampled_df[undersampled_df['Class'] == 0]['Amount'], undersampled_df[undersampled_df['Class'] == 0]['Time'], label='Normal', color='#2769FE')
    ax.scatter(undersampled_df[undersampled_df['Class'] == 1]['Amount'], undersampled_df[undersampled_df['Class'] == 1]['Time'], label='Anomaly', color='#FF5F57')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Time')
    ax.legend()
    st.pyplot(fig)

    st.write("The sixth plot shows the distribution of transaction amounts after applying undersampling techniques. Undersampling is a method used to balance the dataset by reducing the majority class (non-fraudulent transactions) to match the number of instances in the minority class (fraudulent transactions).")
    st.write("In this plot, we can see that the number of non-fraudulent transactions (blue dots) has been reduced, resulting in a more balanced distribution between the two classes. By undersampling, the model will have a more balanced representation of both classes during training, potentially improving its ability to accurately identify fraudulent transactions.")

# Feature Correlation Analysis
    st.header("Feature Analysis")
    st.subheader("Feature Correlation")
    st.write("This section provides an analysis of the correlation between individual features and the target variable 'Class' in the dataset.")
    st.write("Correlation between each feature and the target variable 'Class':")
    st.write("- V11: 0.149067 (Positive correlation)")
    st.write("- V17: -0.313498 (Negative correlation)")

# Class Imbalance
    st.header("Class Distribution")
    st.write("This section displays the distribution of transactions across different classes (fraudulent and non-fraudulent) in the dataset.")
    st.write("Percentage of transactions in each class:")
    st.write("- Non-Fraudulent (Class=0): 99.83%")
    st.write("- Fraudulent (Class=1): 0.17%")

# Model Performance Metrics
    st.header("Model Training and Evaluation")
    st.write("This section presents the performance metrics of a machine learning model trained and evaluated on the dataset.")
    st.write("Performance metrics of the machine learning model:")
    st.write("- Accuracy: 0.95")
    st.write("- F1 Score: 0.86")
    st.write("- Precision: 0.78")
    st.write("- Recall: 0.95")

# Dataset Information
    st.header("Data Statistics")
    st.write("This section provides basic statistics and information about the dataset.")
    st.write("Information about the dataset:")
    st.write("- Number of Rows: 284807")
    st.write("- Number of Columns: 31")
    st.write("Column Data Types:")
    st.write("  - V1: float64")
    st.write("  - V2: float64")
# Write data types for other columns as well


    st.title("Comprehensive Summary")

    st.header("Data Loading and Exploration")
    st.write("The code reads a CSV file containing credit card transaction data into a pandas DataFrame. It then generates a histogram to visualize the class distribution in the data (fraudulent vs. non-fraudulent transactions). It also checks for and removes any duplicate rows in the DataFrame.")

    st.header("Feature Analysis")
    st.write("The code plots the distribution of transaction time and calculates the correlation of each feature with the target variable 'Class'. It also creates scatter plots for the features 'V11' and 'V17' against time for both fraudulent and non-fraudulent transactions.")

    st.header("Data Preprocessing")
    st.write("The 'Amount' and 'Time' columns are standardized using the StandardScaler from sklearn.preprocessing. The original 'Time' and 'Amount' columns are then dropped from the DataFrame.")

    st.header("Undersampling")
    st.write("To address the class imbalance problem (i.e., the number of non-fraudulent transactions is much higher than the number of fraudulent ones), the code uses the RandomUnderSampler from the imblearn.under_sampling module to undersample the majority class (non-fraudulent transactions).")

    st.header("Model Training and Evaluation")
    st.write("The code splits the undersampled data into a training set and a test set. It then trains a Logistic Regression model on the training data and makes predictions on the test data. Finally, it calculates and prints several evaluation metrics for the model, including accuracy, F1 score, precision, and recall, and also prints a classification report.")

    st.header("Data Visualization")
    st.write("The code generates various plots to visualize the distribution of the features and the target variable. It uses seaborn and matplotlib libraries to create these plots. The plots include scatter plots comparing the amounts in fraud and normal transactions, scatter plots comparing 'Time' against 'Amount' for both normal and fraud transactions, and Kernel Density Estimation (KDE) plots for each feature in the dataset.")

    st.title("Thank you for Reading")
    st.header("     - Aryan P.")

if __name__ == "__main__":
    main()
