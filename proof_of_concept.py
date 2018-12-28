# Importing Library
import pandas as pd
import numpy as np

adult_df = pd.read_csv('Dane\\Adult.csv', header=None,
                       delimiter=' *, *', engine='python')  # delimiter=Seperation
# Reading the File
adult_df.head()  # Top 5 Values
adult_df.shape  # Dimensions

# header is not avaliable so we create header
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

# Numerical missing Values are shown as 'NAN'
# categorical missing Values are shown as '?'
adult_df.head()
adult_df.isnull().sum()  # For counting missig values
adult_df.info()  # Data types

# only for categorical data
for value in ['workclass', 'education', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'native_country', 'income']:
    print(value, sum(adult_df[value] == "?"))

# craete a copy of the dataframe
adult_df_rev = pd.DataFrame.copy(adult_df)

# To see all columns
adult_df_rev.describe(include='all')
pd.set_option('display.max_columns', None)

# Missing Values replace by Mode(Catagorical)
for value in ['workclass', 'occupation', 'native_country']:
    adult_df_rev[value].replace(['?'], adult_df_rev[value].mode()[0],
                                inplace=True)  # inplace = replace in original

# check the missing values
for value in ['workclass', 'education', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'native_country', 'income']:
    print(value, sum(adult_df_rev[value] == "?"))

# To find Levels in the categories veriable
adult_df_rev.education.value_counts()

# For preprocessing the data couverting the Catagocical data into Numerical(Lebal Encoding)
colname = ['workclass', 'education', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'native_country', 'income']

from sklearn import preprocessing

le = {}
type(le)
for x in colname:
    le[x] = preprocessing.LabelEncoder()

for x in colname:
    adult_df_rev[x] = le[x].fit_transform(adult_df_rev.__getattr__(x))

adult_df_rev.head()

# Creating to Y and X variable
Y = adult_df_rev.values[:, -1]
Y
X = adult_df_rev.values[:, :-1]
X

# standardizing the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# precautionary step
Y = Y.astype(int)

# Spliting data into Train & Test data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

classifer = (LogisticRegression())
classifer.fit(X_train, Y_train)

# Predicting values
Y_pred = classifer.predict(X_test)
print(list(zip(Y_test, Y_pred)))

# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report

# Confuion MAtrix
cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(Y_test, Y_pred))

# Accuracy of Model = 82.27
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy of model: ", acc)

# Using kfold_cross_validation
classifier = (LogisticRegression())
import sklearn.model_selection as cross_validation

kfold_cv = cross_validation.KFold(n_splits=10)
print(kfold_cv)

# Run the model using scoring metric as accuracy
try:
    kfold_cv_result = cross_validation.cross_val_score(estimator=classifier, X=X_train,
                                                   y=Y_train, cv=kfold_cv)
except FutureWarning:
    pass

print(kfold_cv_result)

# finding the mean Acc = 82.43
print(kfold_cv_result.mean())
