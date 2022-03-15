import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# practice notebook
# https://github.com/MicrosoftDocs/ml-basics/blob/master/challenges/02%20-%20Real%20Estate%20Regression%20Challenge.ipynb

# load data
# dataset link: https://archive-beta.ics.uci.edu/ml/datasets/real+estate+valuation+data+set
house_data = pd.read_csv('real_estate_data.csv')
house_data.describe()

# any nulls? Nope
house_data[house_data.isnull().any(axis=1)]

# plot the scatter plots with the target col
for col in house_data.columns:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = house_data[col]
    label = house_data['Y house price of unit area']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y = label)
    plt.xlabel(col)
    plt.ylabel('House Price')
    ax.set_title('price vs ' + col + '- correlation')
plt.show()


# Separate features from label
X, y = house_data[[
    'X1 transaction date',
    'X2 house age',
    'X3 distance to the nearest MRT station',
    'X4 number of convenience stores',
    'X5 latitude', 
    'X6 longitude']].values, house_data['Y house price of unit area'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# create a model evaluation function
def eval_model(model):
        
    print (model, "\n")

    # Evaluate the model using the test data
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    r2 = r2_score(y_test, predictions)
    print("R2:", r2)

    # Plot predicted vs actual
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('House Price Predictions')
    # overlay the regression line
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    plt.show()

model = LinearRegression().fit(X_train, y_train)
eval_model(model)

model = RandomForestRegressor().fit(X_train, y_train)
eval_model(model)

model = GradientBoostingRegressor().fit(X_train, y_train)
eval_model(model)

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [0,1,2,3,4,5])])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())])

model = pipeline.fit(X_train, y_train)
eval_model(model)