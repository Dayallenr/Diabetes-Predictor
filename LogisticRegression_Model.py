import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler   
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv('/Users/dayallenragunathan/VScode/Diabetes Machine Learning Project/diabetes.csv')
df = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']]


train, test = train_test_split(df, test_size = 0.2, stratify= df['Outcome'], random_state=7)
train, valid = train_test_split(train, test_size = 0.25, stratify= train['Outcome'], random_state=7)

'''def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values


    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    if oversample:
        ros = RandomOverSampler(random_state=3)
        X, y = ros.fit_resample(X, y)

    return dataframe.copy(), X, y
'''

X_train, y_train = train[train.columns[:-1]].values, train[train.columns[-1]].values
X_valid, y_valid= valid[valid.columns[:-1]].values, valid[valid.columns[-1]].values
X_test, y_test = test[test.columns[:-1]].values, test[test.columns[-1]].values

ros = RandomOverSampler(random_state=7)
X_train, y_train = ros.fit_resample(X_train, y_train)

pipe = Pipeline([
                ('scaler', StandardScaler()),
                 ('model', LogisticRegression())
                 ])

model = GridSearchCV(estimator=pipe, param_grid={'model__C': [0.01, 0.1, 1, 10, 100], 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear', 'saga'], 'model__max_iter': [100, 500, 1000], 'model__class_weight': [None, 'balanced', {0:1, 1:2}]}, cv = 5)

model.fit(X_train, y_train)

best_params = model.best_params_

y_pred_valid = model.predict(X_valid)
val_score = balanced_accuracy_score(y_valid, y_pred_valid)



print("Best Parameters on Validation:", best_params)



train_valid = pd.concat([train, valid])
X_train_valid, y_train_valid = train_valid[train_valid.columns[:-1]].values, train_valid[train_valid.columns[-1]].values

X_train_valid, y_train_valid = ros.fit_resample(X_train_valid, y_train_valid)

final_model = Pipeline([
                ('scaler', StandardScaler()),
                 ('model', LogisticRegression(
                    C=best_params['model__C'],
                    penalty=best_params['model__penalty'],
                    solver=best_params['model__solver'],
                    max_iter=best_params['model__max_iter'],
                    class_weight=best_params['model__class_weight']
                 ))
                ])
final_model.fit(X_train_valid, y_train_valid)

y_pred_test = final_model.predict(X_test)


print("\nTest Report:")
print(classification_report(y_test, y_pred_test))
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_test))
