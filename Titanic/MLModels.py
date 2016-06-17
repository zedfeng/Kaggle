from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import random
import numpy as np

# train data
train_data_path = os.path.join(os.path.dirname(__file__) + '/Data/train.csv')
train_data = pd.read_csv(train_data_path, header=0, dtype={"Age": np.float64})
# clean useless data
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

# convert data to float
train_data['Sex'] = train_data['Sex'].map({'female': 0, 'male': 1}).astype(float)
train_data['Embarked'] = train_data['Embarked'].map({'Q': 0, 'S': 1, 'C': 2}).astype(float)
# fill missing data
train_data['Survived'] = train_data['Survived'].fillna(0)
train_data['Pclass'] = train_data['Pclass'].fillna(random.randint(0, 2))
train_data['Sex'] = train_data['Sex'].fillna(random.randint(0, 1))
train_data['SibSp'] = train_data['SibSp'].fillna(0)
train_data['Parch'] = train_data['Parch'].fillna(0)
train_data['Embarked'] = train_data['Embarked'].fillna(random.randint(0, 2))
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].dropna().mean())
train_data_median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        train_data_median_ages[i, j] = train_data.loc[
            (train_data['Sex'] == i) & (train_data['Pclass'] == j + 1), 'Age'].dropna().median()
for i in range(0, 2):
    for j in range(0, 3):
        train_data.loc[train_data['Age'].isnull() & (train_data['Sex'] == i) & (train_data['Pclass'] == j + 1), 'Age'] = \
            train_data_median_ages[i, j]

# test data
test_data_path = os.path.join(os.path.dirname(__file__) + '/Data/test.csv')
test_data = pd.read_csv(test_data_path, header=0, dtype={"Age": np.float64})
# clean useless data
test_data.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
# convert data to float
test_data['Sex'] = test_data['Sex'].map({'female': 0, 'male': 1})
test_data['Embarked'] = test_data['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
# fill missing data
test_data['Pclass'] = test_data['Pclass'].fillna(random.randint(0, 2))
test_data['Sex'] = test_data['Sex'].fillna(random.randint(0, 1))
test_data['SibSp'] = test_data['SibSp'].fillna(0)
test_data['Parch'] = test_data['Parch'].fillna(0)
test_data['Embarked'] = test_data['Embarked'].fillna(random.randint(0, 2))
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].dropna().mean())
test_data_median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        test_data_median_ages[i, j] = test_data.loc[
            (test_data['Sex'] == i) & (test_data['Pclass'] == j + 1), 'Age'].dropna().median()
for i in range(0, 2):
    for j in range(0, 3):
        test_data.loc[test_data['Age'].isnull() & (test_data['Sex'] == i) & (test_data['Pclass'] == j + 1), 'Age'] = \
            test_data_median_ages[i, j]
# data processing
x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
x_test = test_data.drop('PassengerId', axis=1).copy()
# random forest python function requires floats for the input variables
# random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_predict = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
result_path = os.path.join(os.path.dirname(__file__) + '/Data/RandomForestModel.csv')
result = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_predict})
result.to_csv(result_path, index=False)
# logistic regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_predict = log_reg.predict(x_test)
log_reg.score(x_train, y_train)
result_path = os.path.join(os.path.dirname(__file__) + '/Data/LogisticRegressionModel.csv')
result = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_predict})
result.to_csv(result_path, index=False)