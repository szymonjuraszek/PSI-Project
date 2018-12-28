from math import ceil

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

# Wczytywanie danych z pliku Adult.csv
original_data = pd.read_csv(
    'Dane/Adult.csv',
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Income"],
        sep=',',
        engine='python',
        na_values="?")
# print(original_data.head())

# Histogram dla poszczegolnych cech
fig = plt.figure(figsize=(20,15))
cols = 5
rows = ceil(float(original_data.shape[1]) / cols)
for i, column in enumerate(original_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if original_data.dtypes[column] == np.object:
        original_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        original_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()

# print((original_data["Country"].value_counts() / original_data.shape[0]).head());

def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# oblicz korelacje i narysuj
encoded_data, _ = number_encode_features(original_data)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()

# print(original_data[["Education", "Education-Num"]].head(15))

# usuwamy ceche Education poniwaz to to samo co Education-num
del original_data["Education"]

# zamiana stringow na wartosci liczbowe (klasyfikacja)
encoded_data, encoders = number_encode_features(original_data)
fig = plt.figure(figsize=(20,15))
cols = 5
rows = ceil(float(encoded_data.shape[1]) / cols)
for i, column in enumerate(encoded_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    encoded_data[column].hist(axes=ax)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
#plt.show()

################################################################################################
# nazwy kolumn(cech) ktore bierzemy pod uwage
colname = ['Workclass', 'Education-Num', 'Martial Status', 'Occupation',
           'Relationship', 'Race', 'Sex', 'Country', 'Income']

from sklearn import preprocessing

le = {}
type(le)
for x in colname:
    le[x] = preprocessing.LabelEncoder()

for x in colname:
    encoded_data[x] = le[x].fit_transform(encoded_data.__getattr__(x))

encoded_data.head()

# Tworzenie zmiennych X i Y
Y = encoded_data.values[:, -1]
Y
X = encoded_data.values[:, :-1]
X=X.astype('float64')

# standaryzacja danych
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

Y = Y.astype(int)

# Podzielenie danych na dane testowe i do nauki
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

# Wyisywanie zbiorow
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

######################################################################################################################

# Regresja logistyczna
def logReg(X_train, X_test, Y_train, Y_test):

    from sklearn.metrics import confusion_matrix, accuracy_score, \
        classification_report
    import sklearn.model_selection as cross_validation

    # Stworzenie obiektu regresji Logistycznej
    classifer = LogisticRegression()

    # trenowanie modelu uzywajac zbioru testowego
    classifer.fit(X_train, Y_train)

    # Przewidywana wartosc
    Y_pred = classifer.predict(X_test)
    print(list(zip(Y_test, Y_pred)))

    cfm = confusion_matrix(Y_test, Y_pred)
    print(cfm)
    print("Classification report: ")
    print(classification_report(Y_test, Y_pred))

    # Dokladnosc modelu = 82.23
    acc = accuracy_score(Y_test, Y_pred)
    print("Accuracy of model: ", acc)

    # Uzycie kfold_cross_validation
    classifier = LogisticRegression()

    kfold_cv = cross_validation.KFold(n_splits=10)
    print(kfold_cv)

    kfold_cv_result = cross_validation.cross_val_score(estimator=classifier, X=X_train,
                                                       y=Y_train, cv=kfold_cv)

    print(kfold_cv_result)
    print(kfold_cv_result.mean())

# Regresja liniowa
def linReg(X_train, X_test, Y_train, Y_test):

    from sklearn.metrics import mean_squared_error, r2_score

    # Stworzenie obiektu regresji Liniowej
    regr = linear_model.LinearRegression()

    # trenowanie modelu uzywajac zbioru testowego
    regr.fit(X_train, Y_train)

    # predykca dla probki testowej
    Y_pred = regr.predict(X_test)

    # Wspolczynniki
    print('Coefficients: \n', regr.coef_)
    # Rzeczywisty blad srednio-kwadratowy
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, Y_pred))
    # Wspolczynnik predykcji (dla 1 lub -1 idealne dopasowanie)
    print('Variance score: %.2f' % r2_score(Y_test, Y_pred))


logReg(X_train, X_test, Y_train, Y_test)
print('#'*40)
print('#'*40)
print('#'*40)
print('#'*40)
linReg(X_train, X_test, Y_train, Y_test)