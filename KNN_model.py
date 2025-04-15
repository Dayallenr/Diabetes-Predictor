import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler   
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split




df = pd.read_csv('/Users/dayallenragunathan/VScode/Diabetes Machine Learning Project/diabetes.csv')
df = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']]

'''Cleaning and Visualizing data'''

'''
print(df.isna().sum())
print(df.duplicated().sum())
print(df.dtypes)
print(df.shape)

for label in df.columns[:-1]:
    plt.scatter(df[df['Outcome'] == 1][label], [1] * len(df[df['Outcome'] == 1]), color = 'blue', label = 'diabetes', alpha = 0.7)
    plt.scatter(df[df['Outcome'] == 0][label], [0] * len(df[df['Outcome'] == 0]), color = 'red', label = 'no diabetes', alpha = 0.7)
    plt.title(label)
    plt.ylabel('Outcome')
    plt.xlabel(label)
    plt.legend()
    plt.show()
'''


#KNN Model

#Making dataset for training, validating and testing

train, test = train_test_split(df, test_size = 0.2, stratify= df['Outcome'], random_state=42)
train, valid = train_test_split(train, test_size = 0.25, stratify= train['Outcome'], random_state=42)


X_train, y_train = train[train.columns[:-1]].values, train[train.columns[-1]].values
X_valid, y_valid= valid[valid.columns[:-1]].values, valid[valid.columns[-1]].values
X_test, y_test = test[test.columns[:-1]].values, test[test.columns[-1]].values

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

print(sum(y_train == 1))
print(sum(y_train == 0))

pipe = Pipeline([
                ('scaler', StandardScaler()),
                 ('model', KNeighborsClassifier())
                 ])

model = GridSearchCV(estimator=pipe, param_grid={'model__n_neighbors': list(range(1, 35)), 'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]}, cv = 5)

model.fit(X_train, y_train)


y_pred_valid = model.predict(X_valid)
val_score = balanced_accuracy_score(y_valid, y_pred_valid)

best_params = model.best_params_

print("Best Parameters on Validation:", best_params)

train_valid = pd.concat([train, valid])
X_train_valid, y_train_valid = train_valid[train_valid.columns[:-1]].values, train_valid[train_valid.columns[-1]].values

X_train_valid, y_train_valid = ros.fit_resample(X_train_valid, y_train_valid)


final_model = Pipeline([
                ('scaler', StandardScaler()),
                 ('model', KNeighborsClassifier(
                    n_neighbors=best_params['model__n_neighbors'],
                    weights=best_params['model__weights'],
                    p=best_params['model__p']
                 ))
                 ])
final_model.fit(X_train_valid, y_train_valid)

y_pred_test = final_model.predict(X_test)


print("\nTest Report:")
print(classification_report(y_test, y_pred_test))
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_test))