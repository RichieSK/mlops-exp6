import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os

df = pd.read_csv('California_Houses.csv')
df.head()
df.dropna(axis=0, inplace=True)

X = df.iloc[:, 1:7]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training complete!")
joblib.dump(model, "model.joblib")

with open("metrics.txt", 'w') as fw:
  fw.write(f"Mean Squared Error of current model is: {mean_squared_error(y_test, y_pred)}")
