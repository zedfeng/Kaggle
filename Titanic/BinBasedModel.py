import numpy as np
import csv
import os

# open data file
train_data_path = os.path.join(os.path.dirname(__file__) + '/Data/train.csv')
train_data_file = open(train_data_path)
train_data_file_object = csv.reader(train_data_file)
# get header
header = next(train_data_file_object)
# get data
train_data = []
for row in train_data_file_object:
    train_data.append(row)
train_data = np.array(train_data)
# data processing
fare_ceiling = 40
bin_size = 10
num_price_bins = int(fare_ceiling / bin_size)
num_classes = len(np.unique(train_data[0::, 2]))
num_genders = len(np.unique(train_data[0::, 4]))
table_survived = np.zeros((num_genders, num_classes, num_price_bins))
for i in range(num_classes):
    for j in range(num_price_bins):
        women_only_stats = train_data[
            (train_data[0::, 4] == 'female') & (train_data[0::, 2].astype(np.float) == i + 1) & (
                train_data[0::, 9].astype(np.float) >= j * bin_size) & (
            train_data[0::, 9].astype(np.float) < (j + 1) * bin_size), 1]
        men_only_stats = train_data[(train_data[0::, 4] == 'male') & (train_data[0::, 2].astype(np.float) == i + 1) & (
            train_data[0::, 9].astype(np.float) >= j * bin_size) & (
                                        train_data[0::, 9].astype(np.float) < (j + 1) * bin_size), 1]
        table_survived[0, i, j] = np.mean(women_only_stats.astype(np.float))
        table_survived[1, i, j] = np.mean(men_only_stats.astype(np.float))
table_survived[table_survived != table_survived] = 0  # deal with nan
table_survived[table_survived < 0.5] = 0
table_survived[table_survived >= 0.5] = 1
print(table_survived)
train_data_file.close()
# make prediction
test_data_path = os.path.join(os.path.dirname(__file__) + '/Data/test.csv')
test_data_file = open(test_data_path)
test_data_file_object = csv.reader(test_data_file)
test_header = next(test_data_file_object)
result_path = os.path.join(os.path.dirname(__file__) + '/Data/BinBasedModel.csv')
result_file = open(result_path, 'w', newline='')
result_file_object = csv.writer(result_file)
result_file_object.writerow(["PassengerId", "Survived"])
for row in test_data_file_object:
    for j in range(num_price_bins):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = num_price_bins - 1
            break
        if j * bin_size <= row[8] < (j + 1) * bin_size:
            bin_fare = j
            break
    if row[3] == 'female':
        result_file_object.writerow([row[0], table_survived[0, float(row[1]) - 1, bin_fare]])
    if row[3] == 'male':
        result_file_object.writerow([row[0], table_survived[1, float(row[1]) - 1, bin_fare]])
test_data_file.close()
result_file.close()
