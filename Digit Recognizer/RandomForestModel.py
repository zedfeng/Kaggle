from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# train data
train_data_path = os.path.dirname(__file__) + '/Data/train.csv'
train_data = pd.read_csv(train_data_path, header=0)
# test data
test_data_path = os.path.dirname(__file__) + '/Data/test.csv'
test_data = pd.read_csv(test_data_path, header=0)
# random forest
x_train = train_data.drop('label', axis=1)
y_train = train_data['label']
x_test = test_data
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(x_train, y_train)
y_predict = random_forest.predict(x_test)
print(random_forest.score(x_train, y_train))
result_path = os.path.dirname(__file__) + '/Data/RandomForestModel.csv'
result = pd.DataFrame({'ImageId': range(1, len(test_data) + 1), 'label': y_predict})
result.to_csv(result_path, index=False)
