"""Goal of the Project:
From class 67 to class 79, you learned the following concepts:

Feature Encoding.
Recursive Feature Elimination (RFE).
Logistic Regression classification using sklearn module.
In this project, you will apply what you have
 learned in class 67 - 79 to achieve the following goals.

Main Goal	Create a Logistic Regression model classification 
model with ideal number of features selected using RFE."""



"""Problem Statement
The dataset is extracted from 1994 Census Bureau. The data includes an instance of anonymous individual records with features like work-experience, age, gender, country, and so on. Also have divided the records into two labels with people having a salary more than 50K or less than equal to 50K so that they can determine the eligibility of individuals for government opted programs.

Looks like a very interesting dataset and as a data scientist, your job is to build a prediction model to predict whether a particular individual has an annual income of <=50k or >50k.

Things To Do:

Importing and Analysing the Dataset

Data Cleaning

Feature Engineering

Train-Test Split

Data Standardisation

Logistic Regression - Model Training

Model Prediction and Evaluation

Features Selection Using RFE"""


"""ataset Description
The dataset includes 32561 instances with 14 features and 1 target column which can be briefed as:

Field	Description
age	age of the person, Integer
work-class	employment information about the individual, Categorical
fnlwgt	unknown weights, Integer
education	highest level of education obtained, Categorical
education-years	number of years of education, Integer
marital-status	marital status of the person, Categorical
occupation	job title, Categorical
relationship	individual relation in the family-like wife, husband, and so on. Categorical
race	Categorical
sex	gender, Male or Female
capital-gain	gain from sources other than salary/wages, Integer
capital-loss	loss from sources other than salary/wages, Integer
hours-per-week	hours worked per week, Integer
native-country	name of the native country, Categorical
income-group	annual income, Categorical, <=50k or >50k
Notes:

The dataset has no header row for the column name. (Can add column names manually)
There are invalid values in the dataset marked as "?".
As the information about fnlwgt is non-existent it can be removed before model training.
Take note of the whitespaces (" ") throughout the dataset.
Dataset Credits: https://archive.ics.uci.edu/ml/datasets/adult

Dataset Creater:

Dua, D., & Graff, C.. (2017). UCI Machine Learning Repository."""


"""Activity 1: Importing and Analysing the Dataset
In this activity, we have to load the dataset and analyse it.

Perform the following tasks:

Load the dataset into a DataFrame.
Rename the columns with the given list.
Verify the number of rows and columns.
Print the information of the DataFrame.
1. Start with importing all the required modules:

Note: Also import the warnings module and include warnings.filterwarnings('ignore') 
to skip the unnecessary warnings."""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snsort
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

"""2. Create a Pandas DataFrame for the Adult Income dataset using the below link with header=None.

Dataset Link: https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/adult.csv

3. Print the first five rows of the dataset:"""

# Load the Adult Income dataset into DataFrame.
df = pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/adult.csv',header = None)
df.head()

"""Hint: In read_csv() function, header=None parameter allows the creation 
of DataFrame with first row of the file as first row 
rather than column names and the column names are assigned by the system from 0 to n.

4. Rename the columns by applying the rename() function using the following column list:"""

# Rename the column names in the DataFrame using the list given above.
# Create the list
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-years',
            'marital-status', 'occupation', 'relationship', 'race','sex','capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'income-group']

# Rename the columns using 'rename()'
df.rename(columns={0:'age',
                   1:'workclass',
                   2:'fnlwgt',
                   3:'education',
                   4:'education-years',
                   5:'marital-status',
                   6:'occupation',
                   7:'relationship',
                   8: 'race',
                   9:'sex',
                   10:'capital-gain',
                   11:'capital-loss',
                   12:'hours-per-week',
                   13:'native-country',
                   14:'income-group'},inplace = True)
# Print the first five rows of the DataFrame
df.head()

# Print the number of rows and columns of the DataFrame
print("Rows & columns :",df.shape)

# Get the information of the DataFrame
df.info()

# Check the distribution of the labels in the target column.
df['income-group'].value_counts()


"""Activity 2: Data Cleaning
In this activity, we need to clean the DataFrame step by step.

Perform the following tasks:

Check for the null or missing values in the DataFrame.
Observe the categories in column native-country, workclass, and occupation.
Replace the invalid " ?" values in the columns with np.nan using replace() function.
Drop the rows having nan values using the dropna() function.
1. Verify the missing values in the DataFrame:"""

# Check for null values in the DataFrame.
df.dropna().sum()

df.isnull().sum()

# Print the distribution of the columns mentioned to find the invalid values.
# Print the categories in column 'native-country'
print('Unique categories in the column native-country:',df['native-country'].unique())
print('-'*90)

# Print the categories in column 'workclass'
print('Unique categories in the column native-country:',df['workclass'].unique())
print('-'*90)

# Print the categories in column 'occupation'
print('Unique categories in the column native-country:',df['occupation'].unique())

# Replace the invalid values ' ?' with 'np.nan'.
df['native-country'] = df['native-country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)

# Check for null values in the DataFrame again.
df.isnull().sum()

# Delete the rows with invalid values and the column not required
# Delete the rows with the 'dropna()' function
df.dropna(inplace=True)

# Delete the column with the 'drop()' function
df.drop(columns = 'fnlwgt', axis=1, inplace=True)

# Print the number of rows and columns in the DataFrame.
df.shape

"""Activity 3: Feature Engineering
The dataset contains certain features that are categorical. 
To convert these features into numerical ones, use the map() and get_dummies() function.

Perform the following tasks for feature engineering:

Create a list of numerical columns.

Map the values of the column gender to:

Male: 0
Female: 1
Map the values of the column income-group to:

<=50K: 0
>50K: 1
Create a list of categorical columns.

Perform one-hot encoding to obtain numeric values for the rest of the categorical columns.

1. Separate the numeric columns first for that create a list of
 numeric columns using select_dtypes() function:"""

# Create a list of numeric columns names using 'select_dtypes()'.
numeric_df = df.select_dtypes(include='int64')
print(numeric_df.head())
numeric_columns = list(df.select_dtypes(include = ['int64','float64']).columns)
numeric_columns

# Map the 'sex' column and verify the distribution of labels.
# Print the distribution before mapping
print(f"Before mapping \n{df['sex'].value_counts()}")
print('-'*70)


# Map the values of the column to convert the categorical values to integer
df['sex'] = df['sex'].map({' Male':0,' Female':1})
# Print the distribution after mapping
print(f"After mapping \n{df['sex'].value_counts()}")

# Map the 'income-group' column and verify the distribution of labels.
# Print the distribution before mapping
print(f"Before mapping \n{df['income-group'].value_counts()}")
print('-'*70)

# Map the values of the column to convert the categorical values to integer
df['income-group'] = df['income-group'].map({' <=50K' : 0, ' >50K' : 1})
# Print the distribution after mapping
print(f'After mapping \n{df["income-group"].value_counts()}')

# Create the list of categorical columns names using 'select_dtypes()'.
categorical_col=list(df.select_dtypes(include=['object']).columns)
print(categorical_col)


# Create a 'income_dummies_df' DataFrame using the 'get_dummies()' function on the non-numeric categorical columns
income_dummies_df = pd.get_dummies(df[categorical_col], drop_first=True, dtype=int)
income_dummies_df

# Drop the categorical columns from the Income DataFrame `income_df`
income_df = df.drop(columns=df[categorical_col], axis=1)
income_df


# Concat the income DataFrame and dummy DataFrame using 'concat()' function
new_df = pd.concat([income_df, income_dummies_df], axis=1)
new_df.head()

# Get the information of the DataFrame
new_df.info()

new_df.shape

new_df.dtypes

"""Activity 4: Train-Test Split
We need to predict the value of the income-group variable, 
using other variables. Thus, income-group is the target or 
dependent variable and other columns except income-group 
are the features or the independent variables.

1. Split the dataset into the training set and test set
 such that the training set contains 70% of the instances 
 and the remaining instances will become the test set.

2. Set random_state = 42:"""

# Split the training and testing data
# Import the module
X = new_df.drop(columns='income-group',axis = 1)
y = new_df['income-group']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33, random_state = 42)

"""Activity 5: Data Standardisation
To avoid ConvergenceWarning message - That is to scale
 the data using one of the normalisation methods, for instance, standard normalisation.

1. Create a function standard_scalar() to normalise the numeric columns of X_train and
 X_test data-frames using the standard normalisation method:"""

# Normalise the train and test data-frames using the standard normalisation method.
# Define the 'standard_scalar()' function for calculating Z-scores
def standard_scalar(series) :
  return (series - series.mean()) / series.std()

# Create the DataFrames norm_X_train and norm_X_train
norm_X_train = X_train
norm_X_test = X_test
# Apply the 'standard_scalar()' on X_train on numeric columns using apply() function and get the descriptive statistics of the normalised X_train
norm_X_train[numeric_columns] = X_train[numeric_columns].apply(standard_scalar, axis = 0)
print('Train set\n', norm_X_train.describe())
print('-'*70)

# Apply the 'standard_scalar()' on X_test on numeric columns using apply() function and get the descriptive statistics of the normalised X_test
norm_X_test[numeric_columns] = X_test[numeric_columns].apply(standard_scalar, axis = 0)
print('Test set\n', norm_X_test.describe())

"""Activity 6: Logistic Regression - Model Training
Implement Logistic Regression Classification using
 sklearn module in the following way:

Deploy the model by importing the LogisticRegression class 
and create an object of this class.
Call the fit() function on the Logistic Regression object 
and print the score using the score() function."""

# Deploy the 'LogisticRegression' model using the 'fit()' function.
lg_clf = LogisticRegression()
lg_clf.fit(norm_X_train, y_train)
lg_clf.score(norm_X_train, y_train)

"""Activity 7: Model Prediction and Evaluation
1. Predict the values for both training and test sets by calling the predict() function
 on the Logistic Regression object:"""

# Make predictions on the test dataset by using the 'predict()' function.
y_train_pred = lg_clf.predict(norm_X_train)
y_test_pred = lg_clf.predict(norm_X_test)

# Display the results of confusion_matrix
print('Train set\n', confusion_matrix(y_train, y_train_pred))
print('-'*70)
print('Test set\n', confusion_matrix(y_test, y_test_pred))

# Display the results of classification_report
print('Train set\n', classification_report(y_train, y_train_pred))
print('-'*70)
print('Test set\n', classification_report(y_test, y_test_pred))

"""Activity 8: Features Selection Using RFE
Select the relevant features from all the features that contribute the most to classifying individuals in income-groups using RFE.

Steps:

1. Create an empty dictionary and store it in a variable.

2. Create a for loop that iterates through the first 10 columns in the normalised training data-frame. Inside the loop:

Create an object of the Logistic Regression class and store it in a variable.

Create an object of RFE class and store it in a variable. Inside the RFE class constructor, pass the object of logistic regression and the number of features to be selected by RFE as inputs.

Train the model using the fit() function of the RFE class to train a logistic regression model on the train set with i number of features where i goes from 1 to 10 columns in the training dataset.

Create a list to store the important features using the support_ attribute.

Create a new data-frame having the features selected by RFE store in a variable.

Create another Logistic Regression object, store it in a variable and build a logistic regression model using the new training DataFrame created using the rfe features data-frame and the target series.

Predict the target values for the normalised test set (containing the feature(s) selected by RFE) by calling the predict() function on the recent model object.

Calculate f1-scores using the function f1_score() function of sklearn.metrics module that returns a NumPy array containing f1-scores for both the classes. Store the array in a variable called f1_scores_array.

The sytax for the f1_score() is given as:

Syntax: f1_score(y_true, y_pred, average = None)

Where,

a. y_true: the actual labels

b. y_pred: the predicted labels

c. average = None: parameter returns the scores for each class.

Add the number of selected features and the corresponding features 
& f1-scores as key-value pairs in the dictionary.
Note:
As the number of features is very high, the code will be a computationally
 heavy program. It will require very high GPU to process the code faster. 
 It will take some time to learn the feature variables through 
 the training data and then make predictions on the test data.

To turn on the GPU in google colab follow the steps below:

Click on the Edit menu option on the top-left.
Click on the Notebook settings option from the menu. A pop-up will appear.
Click on the drop-down for selecting Hardware accelerator.
Select GPU from the drop-down options.
Click on Save.
"""

# Create a dictionary containing the different combination of features selected by RFE and their corresponding f1-scores.
# Import the libraries
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

# Create the empty dictionary.
dict_rfe = {}
# Create a 'for' loop with range(1,11).
for i in range(1,len(X_train.columns)+1):
  # Create the Logistic Regression Model
    lg_clf2 = LogisticRegression()
  # Create the RFE model with 'i' number of features
    rfe = RFE(lg_clf2)
  # Train the rfe model on the normalised training data using 'fit()'
    rfe.fit(norm_X_train,y_train)

# Create a list of important features chosen by RFE.
    rfe_features = list(norm_X_train.columns[rfe.support_])
  # Create the normalised training DataFrame with rfe features
    rfe_X_train = norm_X_train[rfe_features]
  # Create the logistic regression
    lg_clf3 = LogisticRegression()
  # Train the model normalised training DataFrame with rfe features using 'fit()'
    lg_clf3.fit(rfe_X_train, y_train)
  # Predict 'y' values only for the test set as generally, they are predicted quite accurately for the train set.
    y_test_predict = lg_clf3.predict(norm_X_test[rfe_features])
  # Calculate the f1-score
    f1_scores_array = f1_score(y_test, y_test_predict, average=None)
  # Add the name of features and f1-scores in the dictionary
    dict_rfe[i] = {'features' : list(rfe_features),
                   'f1_score' : f1_scores_array}
    
# Print the dictionary
dict_rfe

# Convert the dictionary to the DataFrame
pd.options.display.max_colwidth = 200
rfe_df = pd.DataFrame.from_dict(dict_rfe)
rfe_df

"""Activity 9: Model Training and Prediction Using Ideal Features
1. Create the logistic regression model again using RFE with the 
ideal number of features and predict the target variable:"""

# Logistic Regression with the ideal number of features and predict the target.

# Create the Logistic Regression Model
lg_clf3 = LogisticRegression()

# Create the RFE model with ideal number of features
rfe2 = RFE(lg_clf3, 4)
# Train the rfe model on the normalised training data
rfe2.fit(norm_X_train, y_train)
# Create a list of important features chosen by RFE.
rfe2_features = norm_X_train.columns[rfe2.support_]
# Create the normalised training DataFrame with rfe features
final_X_train = norm_X_train[rfe2_features]
# Create the Regression Model again
lg_clf4 = LogisticRegression()

# Train the model with the normalised training features DataFrame with best rfe features and target training DataFrame
lg_clf4.fit(final_X_train, y_train)

# Predict the target using the normalised test DataFrame with rfe features
y_test_predict = lg_clf4.predict(norm_X_test[rfe2_features])

# Calculate the final f1-score and print it
final_f1_scores_array = f1_score(y_test, y_test_predict, average=None)
print(final_f1_scores_array)

