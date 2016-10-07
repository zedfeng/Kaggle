import pandas as pd
import pylab as pl
import os

# open data file
train_data_path = os.path.dirname(__file__) + '/Data/train.csv'
train_data_path = os.path.dirname(__file__) + '/Data/train.csv'
train_data = pd.read_csv(train_data_path, header=0)
# data processing
train_data['Age'].dropna().hist(bins=16, range=(0, 80), alpha=0.5)
train_data[train_data['Survived'] == 1]['Age'].dropna().hist(bins=16, range=(0, 80), alpha=0.5)
train_data[(train_data['Sex'] == 'female') & (train_data['Survived'] == 1)]['Age'].dropna().hist(bins=16,
                                                                                                 range=(
                                                                                                     0, 80),
                                                                                                 alpha=0.5)
pl.show()
