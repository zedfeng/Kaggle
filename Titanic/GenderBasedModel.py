import numpy as np
import csv
import os

# open data file
test_data_path = os.path.join(os.path.dirname(__file__) + '/Data/test.csv')
test_data_file = open(test_data_path)
test_data_file_object = csv.reader(test_data_file)
# get header
header = next(test_data_file_object)
# get data
test_data = []
for row in test_data_file_object:
    test_data.append(row)
test_data = np.array(test_data)
# data processing
result_path = os.path.join(os.path.dirname(__file__) + '/Data/GenderBasedModel.csv')
result_file = open(result_path, 'w', newline='')
result_file_object = csv.writer(result_file)
result_file_object.writerow(["PassengerId", "Survived"])
for row in test_data:
    if row[3] == 'female':
        result_file_object.writerow([row[0], '1'])
    else:
        result_file_object.writerow([row[0], '0'])
test_data_file.close()
result_file.close()
