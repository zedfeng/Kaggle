import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

data_path = os.path.dirname(__file__) + '/Data/wdbc.data'
data = pd.read_csv(data_path, sep=",", header=None)
data.columns = ["id", "class", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                "compactness_mean", "concavity_mean", "concavepoints_mean", "symmetry_mean", "fractaldimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                "concavepoints_se", "symmetry_se", "fractaldimension_se", "radius_worst", "texture_worst",
                "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                "concavepoints_worst", "symmetry_worst", "fractaldimension_worst"]
features = data[data.columns[2:32]]
target = data["class"]

# random forest
random_forest = RandomForestClassifier(n_estimators=1000)
scores = cross_val_score(random_forest, features, target, cv=5)
print(scores)
