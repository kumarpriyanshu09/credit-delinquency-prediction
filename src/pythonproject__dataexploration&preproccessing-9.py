# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm
import math
import seaborn as sns
import statsmodels.api as sm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from scipy.stats import uniform
from io import StringIO
from sklearn.impute import SimpleImputer

myDF0 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Project/archive/application_record (1).csv")
myDF1 = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Project/archive/credit_record.csv")

print(myDF0.columns)
print(myDF1.columns)

myDF0.head()

myDF1.head()

myDF0.info()

myDF1.info()

myDF0.describe()

myDF1.describe()

# Example: Aggregating multiple credit records for each ID
agg_rules = {
    'MONTHS_BALANCE': ['min', 'max', 'mean'],  # Numeric column example
    'STATUS': ['max']  # Categorical column example, assuming 'STATUS' indicates delinquency level
}
credit_summary = myDF1.groupby('ID').agg(agg_rules).reset_index()
# Flatten MultiIndex in columns
credit_summary.columns = ['_'.join(col).strip() for col in credit_summary.columns.values]

print(credit_summary.columns)

credit_summary.head()

# Renaming 'ID_' to 'ID' for consistency before merging
credit_summary.rename(columns={'ID_': 'ID'}, inplace=True)

myDF2 = myDF0.merge(credit_summary, how = 'left', on = ['ID'])

print(myDF2.columns)

myDF2.isnull().sum()

myDF2 = myDF2.dropna()

myDF2.isnull().sum()

myDF2.describe()

myDF2.info()

size_mapping = {
'5': 7,
'4': 6,
'3': 5,
'2': 4,
'1': 3,
'0': 2,
'X': 1,
'C': 0
}
myDF2['STATUS_max'] = myDF2['STATUS_max'].map(size_mapping)

myDF2['Proxy_Late_status'] = (myDF2['STATUS_max'] >= 3).astype(int)

X_categorical = pd.get_dummies(myDF2[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']])
X_numerical = myDF2[['CNT_CHILDREN', 'AMT_INCOME_TOTAL']]

X = pd.concat([X_numerical, X_categorical], axis=1)
y = myDF2.Proxy_Late_status

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
feat_labels = X.columns
forest = RandomForestClassifier(n_estimators=10,
                                random_state=0,
                                n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Slide 50: Printing out features
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

print(X.columns)

X.info()

print('Application Record data shape: ',X.shape)
print('Credit Record data shape: ',y.shape)
print('Merged data shape: ',myDF2.shape)

myDF2.head()

myDF2.duplicated().sum()

numerical_col=myDF2.select_dtypes('number').columns
numerical_coldf=myDF2.select_dtypes('number').columns
plt.figure(figsize=(10,10))
sns.heatmap(myDF2[numerical_col].corr(),cmap='rocket',fmt='.2f',annot=True,vmin=-1,vmax=1)

#Piechart
plt.figure(figsize=(6,6))
colors = ['#ff9999', '#66b3ff']
explode = (0, 0.1)
plt.pie(x=myDF2['CODE_GENDER'].value_counts().values, labels=['Female', 'Male'],
        autopct='%.1f%%', startangle=90, explode=explode, colors=colors, shadow=True)
plt.legend(title="Gender", loc="upper right")
plt.title("Gender Distribution")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'FLAG_OWN_CAR' is present in both myDF0 and myDF2 and you want to use myDF0 for this plot
# First, calculate the value counts
car_ownership_counts = myDF0['FLAG_OWN_CAR'].value_counts().reset_index()

# Rename columns for clarity
car_ownership_counts.columns = ['Owns_Car', 'Count']

# Now, plot
plt.figure(figsize=(6,6))
ax = sns.barplot(data=car_ownership_counts, y='Count', x='Owns_Car', palette='rocket')

total = len(myDF0)  # Total counts

# Adding bar labels
for p in ax.patches:
    height = p.get_height()
    percentage = f'{100 * height / total:.0f}%'  # Calculate percentage
    ax.annotate(percentage,
                (p.get_x() + p.get_width() / 2., height),  # Position for annotation
                ha='center', va='bottom',  # Alignment
                fontsize=12, color='black', fontweight='bold')  # Styling

# Customize x and y axis labels
ax.set_xlabel("Owns Car")
ax.set_ylabel("Count")

# Add a title
ax.set_title("Distribution of Car Ownership")

plt.show()

#Histogram
plt.figure(figsize=(10, 6))

ax = sns.histplot(data=myDF2[myDF2['AMT_INCOME_TOTAL'] <= 600000], x='AMT_INCOME_TOTAL', bins=30,
                  color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Income')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Income')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

#Barchart (count)
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate value counts outside the plotting command for clarity
occupation_counts = myDF2['OCCUPATION_TYPE'].value_counts()

plt.figure(figsize=(10, 8))

# Use ax for plotting to make it easier to add annotations later
ax = sns.barplot(x=occupation_counts.values,
                 y=occupation_counts.index,
                 palette='rocket')

# Iterate over the values to place the labels (no need to access app_rec here)
for i, v in enumerate(occupation_counts.values):
    ax.text(v + 3, i, str(v), color='black', va='center')

plt.xlabel('Count')
plt.ylabel('Occupation Type')
plt.title('Distribution of Occupation Types')

plt.show()

# Convert 'OCCUPATION_TYPE' to dummy variables
occupation_dummies = pd.get_dummies(myDF2['OCCUPATION_TYPE'])

# Choose a method of correlating these with 'STATUS', for example, taking the mean of 'STATUS' for each category
for column in occupation_dummies.columns:
    print(f"Correlation between STATUS and {column}: {occupation_dummies[column].corr(myDF2['STATUS_max'])}")

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Assuming df is your DataFrame containing the two columns
# Convert the categorical variables to numerical labels
myDF2['NAME_HOUSING_TYPE'] = pd.Categorical(myDF2['NAME_HOUSING_TYPE']).codes
myDF2['FLAG_OWN_REALTY'] = pd.Categorical(myDF2['FLAG_OWN_REALTY']).codes

# Create a contingency table
contingency_table = pd.crosstab(myDF2['NAME_HOUSING_TYPE'], myDF2['FLAG_OWN_REALTY'])

# Calculate Cramer's V
def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

cramers_v_statistic = cramers_v(contingency_table)
print("Cramer's V statistic:", cramers_v_statistic)

myDF2.MONTHS_BALANCE_mean.unique()

myDF2.MONTHS_BALANCE_mean.describe()

# Define a function for custom binning that returns numeric codes
def custom_bin_numeric(months_balance):
    if months_balance <= -60:
        return 4  # "13.5+ Months Past"
    elif months_balance > -13.5 and months_balance <= -23:
        return 3  # "6-13.5 Months Past"
    elif months_balance > -6 and months_balance <= -13.5:
        return 2  # "1-5 Months Past"
    elif months_balance < -6:
        return 1  # "Current Month"
    else:
        return 0  # Assuming positive values are unexpected or represent a default group

# Apply custom binning
myDF2['Balance_Numeric_Category'] = myDF2['MONTHS_BALANCE_mean'].apply(custom_bin_numeric)

print(myDF2)

myDF2

pip install statsmodels

# Creating a new column based on conditions directly with pandas
myDF2['Target_Variable'] = ((myDF2['Proxy_Late_status'] == 1) & (myDF2['MONTHS_BALANCE_mean'] <= 2)).astype(int)

X = pd.concat([X_numerical, X_categorical], axis=1)
y = myDF2.Target_Variable

# Count the frequency of STATUS_max > 5
frequency = sum(1 for entry in myDF2["STATUS_max"] if entry > -1)

print("Frequency of STATUS_max > 5:", frequency)

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure only numeric columns are included and exclude the Target_Variable
numeric_cols = myDF2.select_dtypes(include=[np.number]).drop(columns=['Target_Variable'], errors='ignore')  # Use errors='ignore' to avoid issues if the column does not exist

# Calculate the correlation matrix
correlation_matrix = numeric_cols.corr()

# Now add the Target_Variable to view its correlation with other features
if 'Target_Variable' in myDF2.columns:
    # Calculate correlation with Target_Variable separately to include in the plot
    target_correlation = myDF2.corr()['Target_Variable'].drop('Target_Variable')  # Drop the target's own correlation entry

    # Plotting the correlation values
    plt.figure(figsize=(10, 8))
    sns.barplot(x=target_correlation.values, y=target_correlation.index)
    plt.title('Feature Correlation with Target_Variable')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.show()

    # Output the sorted correlation values
    print(target_correlation)
else:
    print("Column 'Target_Variable' does not exist in DataFrame.")

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Ensure all columns are numeric and convert them explicitly to float to avoid data type issues
X = pd.get_dummies(myDF2.drop(['Target_Variable'], axis=1), drop_first=True).astype(float)

# Handling any infinite or NaN values in X
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Add a constant term for the intercept (required for VIF calculation)
X_with_constant = add_constant(X)

# Initialize DataFrame to store VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_constant.columns

# Calculate VIF for each feature using a list comprehension to ensure all values are finite
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i)
                   for i in range(X_with_constant.shape[1])]

# Display the VIF
print(vif_data.sort_values(by='VIF', ascending=False))

