from sklearn import svm
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
clf = svm.SVC(kernel='poly')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
result_path = os.path.dirname(__file__) + '/Data/SVMModel.csv'
result = pd.DataFrame({'ImageId': range(1, len(test_data) + 1), 'label': y_predict})
result.to_csv(result_path, index=False)
