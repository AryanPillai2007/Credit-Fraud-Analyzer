import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# First Portion

df1 = pd.read_csv('/Users/aryanpillai2701/Library/On Disk/Files/Credit-Fraud-Analyzer/creditcard.csv')
df1.head(5)

count_classes = pd.Series(df1['Class']).value_counts(sort=True).sort_index()
count_classes.plot(kind = 'bar', color=['#1f77b4', '#ff7f0e'])
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.yscale('log')  # Add this line
plt.show()



df1.describe()
df1.duplicated().any()
# Drop the duplicate rows from the DataFrame
df1 = df1.drop_duplicates()
# Reset index
df1 = df1.reset_index(drop=True)

# Checking that all duplicate rows are dropped
df1.duplicated().any()
# Checking if the target('Class') data is balanced
df1.Class.value_counts()
# taking a copy of data
df2 = df1.copy()


fig, ax = plt.subplots(figsize=(10,5))
ax.set_facecolor('lightgray')
ax.plot(df2.index, df2["Time"], color='red')
ax.set_xlabel("Transaction")
ax.set_ylabel("Time")
ax.set_title("Distribution of Transaction Time")

plt.show()


corr = df2.corr()
c = corr['Class'].sort_values(ascending=False)
print(c)

# value of 1 in the "Class" column indicates (fraud) transactions
fraud = df2[df2["Class"] == 1]
# value of 0 in the "Class" column indicates (real) transactions
real = df2[df2["Class"] == 0]

# V11 Graph
plt.figure(figsize=(10, 5))
plt.scatter(real["Time"], real["V11"], label="Real Transaction")
plt.scatter(fraud["Time"], fraud["V11"], label="Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("V11 class")
plt.title("Max positive correlation (V11)")
plt.legend()
plt.show()

# V17 Graph
plt.figure(figsize=(10, 5))
plt.scatter(real["Time"], real["V17"], label="Real Transaction")
plt.scatter(fraud["Time"], fraud["V17"], label="Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("V17 class")
plt.title("Max negative correlation (V17)")
plt.legend()
plt.show()

# standardizes the 'Amount' column in the DataFrame
df2['amount'] = StandardScaler().fit_transform(df2['Amount'].values.reshape(-1, 1))
# standardizes the 'Tine' column in the DataFrame
df2['time'] = StandardScaler().fit_transform(df2['Time'].values.reshape(-1, 1))
# Removng this non-scaling columns
df2.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df2.drop('Class', axis=1)
Y = df2['Class']

undersampler = RandomUnderSampler(random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X_undersampled, y_undersampled, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Percentage of fraud/no fraud
print('No Frauds', round(df1['Class'].value_counts()[0]/len(df1) * 100,2), '% of the dataset')
print('Frauds', round(df1['Class'].value_counts()[1]/len(df1) * 100,2), '% of the dataset')

# Evaluate the model with  accuracy, F1 score, precision and recall
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
classifcation_report_logistic = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classifcation_report_logistic)



# Second Portion

background_color = '#FFFFFF'
color_palette=['#2769FE', '#FF5F57', '#4dad82', '#230F88', '#0E0330']

df = pd.read_csv('/Users/aryanpillai2701/Library/On Disk/Files/Credit-Fraud-Analyzer/creditcard.csv')
df.info()

df.describe()
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]
fraud['Amount'].describe()

normal['Amount'].describe()

fig = plt.figure(figsize=(8, 5))

fig.set_facecolor(background_color)
plt.gca().set_facecolor(background_color)
ax = plt.gca()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
    
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.plot(normal['Amount'], label='Normal', color=color_palette[0])
plt.plot(fraud['Amount'], label='Fraud', color=color_palette[1])

plt.title('Amount in Fraud vs Normal Transactions')
plt.xlabel('Index')
plt.ylabel('Amount')
plt.legend(loc='upper right')

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

sns.scatterplot(data=normal, x='Amount', y='Time', color=color_palette[0], label='Normal', ax=ax)
sns.scatterplot(data=fraud, x='Amount', y='Time', color=color_palette[1], label='Anomaly', ax=ax)

ax.set_xlabel('Amount', fontweight='bold', fontfamily='serif')
ax.set_ylabel('Time', fontweight='bold', fontfamily='serif')
ax.legend()

ax.tick_params(left=True, bottom=True)
ax.grid(True, color='#000000', linestyle=':', alpha=0.5)

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)

plt.show()

new_df = pd.concat([fraud, normal.sample(n=len(fraud))], axis=0)
new_df = new_df.sample(frac=1).reset_index(drop=True)
new_df['Class'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

sns.scatterplot(data=new_df[new_df['Class'] == 1], x='Amount', y='Time', color=color_palette[0], label='Normal', ax=ax)
sns.scatterplot(data=new_df[new_df['Class'] == 0], x='Amount', y='Time', color=color_palette[1], label='Anomaly', ax=ax)

ax.set_xlabel('Amount', fontweight='bold', fontfamily='serif')
ax.set_ylabel('Time', fontweight='bold', fontfamily='serif')
ax.legend()

ax.tick_params(left=True, bottom=True)
ax.grid(True, color='#000000', linestyle=':', alpha=0.5)

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)

plt.show()



