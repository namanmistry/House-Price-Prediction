import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump,load

housing = pd.read_csv("./data/data.csv")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()
housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)

my_pipeline = Pipeline([("imputer",SimpleImputer(strategy='median')),("scaler",StandardScaler())])
housing_num_tr = my_pipeline.fit_transform(housing) 

model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores are: ",scores)
    print("Mean is: ", scores.mean())
    print("Standard deviation: ",scores.std())

dump(model, './model/Dragon.joblib')
print_scores(rmse_scores)