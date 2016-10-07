from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# train data
train_data_path = os.path.dirname(__file__) + '/Data/train.csv'
train_data = pd.read_csv(train_data_path, header=0)
# test data
test_data_path = os.path.dirname(__file__) + '/Data/test.csv'
test_data = pd.read_csv(test_data_path, header=0)
# data processing
