from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# train data
train_data_path = os.path.join(os.path.dirname(__file__) + '/Data/train.csv')
train_data = pd.read_csv(train_data_path, header=0)
print(train_data)
