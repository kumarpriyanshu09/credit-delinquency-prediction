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

myDF0.head()

myDF1.head()

# Example: Aggregating multiple credit records for each ID
agg_rules = {
    'MONTHS_BALANCE': ['min', 'max', 'mean'],  # Numeric column example
    'STATUS': ['max']  # Categorical column example, assuming 'STATUS' indicates delinquency level
}
credit_summary = myDF1.groupby('ID').agg(agg_rules).reset_index()
# Flatten MultiIndex in columns
credit_summary.columns = ['_'.join(col).strip() for col in credit_summary.columns.values]

# Renaming 'ID_' to 'ID' for consistency before merging
credit_summary.rename(columns={'ID_': 'ID'}, inplace=True)

credit_summary.head()

credit_summary.isnull().sum()

myDF2 = myDF0.merge(credit_summary, how = 'left', on = ['ID'])

myDF2.head()

myDF2.describe()

# Calculate IQR for AMT_INCOME_TOTAL
Q1 = myDF2['AMT_INCOME_TOTAL'].quantile(0.25)
Q3 = myDF2['AMT_INCOME_TOTAL'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers and overwrite myDF2 with the filtered dataset
myDF2 = myDF2[(myDF2['AMT_INCOME_TOTAL'] >= lower_bound) & (myDF2['AMT_INCOME_TOTAL'] <= upper_bound)]

# Print the new shape of myDF2 to see how many rows were removed
print('New shape after removing outliers:', myDF2.shape)

myDF2.head()

myDF2.isnull().sum()

print('Application Record data shape: ',myDF0.shape)
print('Credit Record data shape: ',myDF1.shape)
print('Aggregated Credit Record data shape: ',credit_summary.shape)
print('Merged data shape: ',myDF2.shape)

# Define a function for custom binning that returns numeric codes
def custom_bin_numeric(months_balance):
    if months_balance <= -60:
        return 4  # "12+ Months Past"
    elif months_balance > -13.5 and months_balance <= -23:
        return 3  # "6-11 Months Past"
    elif months_balance > -6 and months_balance <= -13.5:
        return 2  # "1-5 Months Past"
    elif months_balance < -6:
        return 1  # "Current Month"
    else:
        return 0  # Assuming positive values are unexpected or represent a default group

# Apply custom binning
myDF2['Balance_Numeric_Category'] = myDF2['MONTHS_BALANCE_mean'].apply(custom_bin_numeric)

print('Merged data shape: ',myDF2.shape)

myDF2.isnull().sum()

myDF2.drop('OCCUPATION_TYPE',axis=1,inplace=True)

myDF2 = myDF2.dropna()
myDF2.isnull().sum()

print('DF2 data shape: ',myDF2.shape)

size_mapping = {
'5': 7,
'4': 6,
'3': 5,
'2': 4,
'1': 3,
'0': 2,
'X': 1,
'C': 0}
myDF2['STATUS_max'] = myDF2['STATUS_max'].map(size_mapping)

myDF2['Proxy_Late_status'] = (myDF2['STATUS_max'] >= 5).astype(int)

# Creating a new column based on conditions directly with pandas
myDF2['Target_Variable'] = ((myDF2['Proxy_Late_status'] == 1) & (myDF2['MONTHS_BALANCE_mean'] <= 2)).astype(int)

# Count the frequency of rows that meet both conditions
frequency = sum((myDF2['Proxy_Late_status'] == 1) & (myDF2['MONTHS_BALANCE_mean'] <= 2))

print("Frequency of rows that meet both conditions:", frequency)

# Count the frequency of rows that do not meet both conditions
frequency = sum(~((myDF2['Proxy_Late_status'] == 1) & (myDF2['MONTHS_BALANCE_mean'] <= 2)))

print("Frequency of rows that do not meet both conditions:", frequency)

myDF2.Target_Variable

X_categorical = pd.get_dummies(myDF2[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_FAMILY_STATUS']])
X_numerical = myDF2[['CNT_CHILDREN', 'AMT_INCOME_TOTAL']]

X = pd.concat([X_numerical, X_categorical], axis=1)
y = myDF2.Target_Variable

X = X.drop(['CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y'], axis=1)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Selecting the family status columns from X
family_status_features = X[['NAME_FAMILY_STATUS_Civil marriage',
                            'NAME_FAMILY_STATUS_Married',
                            'NAME_FAMILY_STATUS_Separated',
                            'NAME_FAMILY_STATUS_Single / not married',
                            'NAME_FAMILY_STATUS_Widow']]

# Standardizing the family status features
scaler = StandardScaler()
family_status_std = scaler.fit_transform(family_status_features)

# Applying PCA to the family status features
pca = PCA(n_components=1)  # Change this if you want more components
family_status_pca = pca.fit_transform(family_status_std)

# Adding the PCA component back to the X DataFrame as a new column
X['Family_Status_PCA'] = family_status_pca

# Now drop the original family status columns from X
X.drop(columns=family_status_features.columns, inplace=True)

X.head()

print('Predictor data shape: ',X.shape)

print('Target data shape: ',y.shape)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Split the original dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Apply oversampling to the training set only
oversampler = RandomOverSampler(random_state=0)
X_train, y_train = oversampler.fit_resample(X_train, y_train)  # Reassign the oversampled data back

# Now, X_train and y_train contain the oversampled training data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Fitting the model
tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=0)
tree.fit(X_train_std, y_train)

# Making predictions and evaluating the model
y_pred_train = tree.predict(X_train_std)
y_pred_test = tree.predict(X_test_std)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize the pipeline with the scaler and classifier
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=0))
])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = []

# Perform the K-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Fit the pipeline on the training part of the fold
    pipe_dt.fit(X_train_fold, y_train_fold)

    # Score the pipeline on the test part of the fold
    score = pipe_dt.score(X_test_fold, y_test_fold)
    scores.append(score)

# Print the performance
print(f'Cross-validation scores: {scores}')
print(f'Average cross-validation score: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

# Use scikit-learn's k-fold cross-validation scorer
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=tree,
								X=X_train,
								y=y_train,
								cv=10,
								n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print(f'Average cross-validation score: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

from sklearn.model_selection import GridSearchCV

# Parameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=0)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train_std, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Use the best estimator to make predictions
y_pred = grid_search.best_estimator_.predict(X_test_std)

# Evaluate the predictions
test_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set: {:.2f}".format(test_accuracy))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Fitting the model
tree = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf = 1, min_samples_split = 2)
tree.fit(X_train_std, y_train)

# Making predictions and evaluating the model
y_pred_train = tree.predict(X_train_std)
y_pred_test = tree.predict(X_test_std)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Testing Accuracy: {test_accuracy:.2f}')

